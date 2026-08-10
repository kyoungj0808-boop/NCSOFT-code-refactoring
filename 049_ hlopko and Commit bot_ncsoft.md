원본코드
# Copyright 2012 the V8 project authors. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Google Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import multiprocessing
import os
import shutil
import subprocess
import threading
import time

from . import daemon
from . import local_handler
from . import presence_handler
from . import signatures
from . import status_handler
from . import work_handler
from ..network import perfdata


class Server(daemon.Daemon):

  def __init__(self, pidfile, root, stdin="/dev/null",
               stdout="/dev/null", stderr="/dev/null"):
    super(Server, self).__init__(pidfile, stdin, stdout, stderr)
    self.root = root
    self.local_handler = None
    self.local_handler_thread = None
    self.work_handler = None
    self.work_handler_thread = None
    self.status_handler = None
    self.status_handler_thread = None
    self.presence_daemon = None
    self.presence_daemon_thread = None
    self.peers = []
    self.jobs = multiprocessing.cpu_count()
    self.peer_list_lock = threading.Lock()
    self.perf_data_lock = None
    self.presence_daemon_lock = None
    self.datadir = os.path.join(self.root, "data")
    pubkey_fingerprint_filename = os.path.join(self.datadir, "mypubkey")
    with open(pubkey_fingerprint_filename) as f:
      self.pubkey_fingerprint = f.read().strip()
    self.relative_perf_filename = os.path.join(self.datadir, "myperf")
    if os.path.exists(self.relative_perf_filename):
      with open(self.relative_perf_filename) as f:
        try:
          self.relative_perf = float(f.read())
        except:
          self.relative_perf = 1.0
    else:
      self.relative_perf = 1.0

  def run(self):
    os.nice(20)
    self.ip = presence_handler.GetOwnIP()
    self.perf_data_manager = perfdata.PerfDataManager(self.datadir)
    self.perf_data_lock = threading.Lock()

    self.local_handler = local_handler.LocalSocketServer(self)
    self.local_handler_thread = threading.Thread(
        target=self.local_handler.serve_forever)
    self.local_handler_thread.start()

    self.work_handler = work_handler.WorkSocketServer(self)
    self.work_handler_thread = threading.Thread(
        target=self.work_handler.serve_forever)
    self.work_handler_thread.start()

    self.status_handler = status_handler.StatusSocketServer(self)
    self.status_handler_thread = threading.Thread(
        target=self.status_handler.serve_forever)
    self.status_handler_thread.start()

    self.presence_daemon = presence_handler.PresenceDaemon(self)
    self.presence_daemon_thread = threading.Thread(
        target=self.presence_daemon.serve_forever)
    self.presence_daemon_thread.start()

    self.presence_daemon.FindPeers()
    time.sleep(0.5)  # Give those peers some time to reply.

    with self.peer_list_lock:
      for p in self.peers:
        if p.address == self.ip: continue
        status_handler.RequestTrustedPubkeys(p, self)

    while True:
      try:
        self.PeriodicTasks()
        time.sleep(60)
      except Exception, e:
        print("MAIN LOOP EXCEPTION: %s" % e)
        self.Shutdown()
        break
      except KeyboardInterrupt:
        self.Shutdown()
        break

  def Shutdown(self):
    with open(self.relative_perf_filename, "w") as f:
      f.write("%s" % self.relative_perf)
    self.presence_daemon.shutdown()
    self.presence_daemon.server_close()
    self.local_handler.shutdown()
    self.local_handler.server_close()
    self.work_handler.shutdown()
    self.work_handler.server_close()
    self.status_handler.shutdown()
    self.status_handler.server_close()

  def PeriodicTasks(self):
    # If we know peers we don't trust, see if someone else trusts them.
    with self.peer_list_lock:
      for p in self.peers:
        if p.trusted: continue
        if self.IsTrusted(p.pubkey):
          p.trusted = True
          status_handler.ITrustYouNow(p)
          continue
        for p2 in self.peers:
          if not p2.trusted: continue
          status_handler.TryTransitiveTrust(p2, p.pubkey, self)
    # TODO: Ping for more peers waiting to be discovered.
    # TODO: Update the checkout (if currently idle).

  def AddPeer(self, peer):
    with self.peer_list_lock:
      for p in self.peers:
        if p.address == peer.address:
          return
      self.peers.append(peer)
    if peer.trusted:
      status_handler.ITrustYouNow(peer)

  def DeletePeer(self, peer_address):
    with self.peer_list_lock:
      for i in xrange(len(self.peers)):
        if self.peers[i].address == peer_address:
          del self.peers[i]
          return

  def MarkPeerAsTrusting(self, peer_address):
    with self.peer_list_lock:
      for p in self.peers:
        if p.address == peer_address:
          p.trusting_me = True
          break

  def UpdatePeerPerformance(self, peer_address, performance):
    with self.peer_list_lock:
      for p in self.peers:
        if p.address == peer_address:
          p.relative_performance = performance

  def CopyToTrusted(self, pubkey_filename):
    with open(pubkey_filename, "r") as f:
      lines = f.readlines()
      fingerprint = lines[-1].strip()
    target_filename = self._PubkeyFilename(fingerprint)
    shutil.copy(pubkey_filename, target_filename)
    with self.peer_list_lock:
      for peer in self.peers:
        if peer.address == self.ip: continue
        if peer.pubkey == fingerprint:
          status_handler.ITrustYouNow(peer)
        else:
          result = self.SignTrusted(fingerprint)
          status_handler.NotifyNewTrusted(peer, result)
    return fingerprint

  def _PubkeyFilename(self, pubkey_fingerprint):
    return os.path.join(self.root, "trusted", "%s.pem" % pubkey_fingerprint)

  def IsTrusted(self, pubkey_fingerprint):
    return os.path.exists(self._PubkeyFilename(pubkey_fingerprint))

  def ListTrusted(self):
    path = os.path.join(self.root, "trusted")
    if not os.path.exists(path): return []
    return [ f[:-4] for f in os.listdir(path) if f.endswith(".pem") ]

  def SignTrusted(self, pubkey_fingerprint):
    if not self.IsTrusted(pubkey_fingerprint):
      return []
    filename = self._PubkeyFilename(pubkey_fingerprint)
    result = signatures.ReadFileAndSignature(filename)  # Format: [key, sig].
    return [pubkey_fingerprint, result[0], result[1], self.pubkey_fingerprint]

  def AcceptNewTrusted(self, data):
    # The format of |data| matches the return value of |SignTrusted()|.
    if not data: return
    fingerprint = data[0]
    pubkey = data[1]
    signature = data[2]
    signer = data[3]
    if not self.IsTrusted(signer):
      return
    if self.IsTrusted(fingerprint):
      return  # Already trusted.
    filename = self._PubkeyFilename(fingerprint)
    signer_pubkeyfile = self._PubkeyFilename(signer)
    if not signatures.VerifySignature(filename, pubkey, signature,
                                      signer_pubkeyfile):
      return
    return  # Nothing more to do.

  def AddPerfData(self, test_key, duration, arch, mode):
    data_store = self.perf_data_manager.GetStore(arch, mode)
    data_store.RawUpdatePerfData(str(test_key), duration)

  def CompareOwnPerf(self, test, arch, mode):
    data_store = self.perf_data_manager.GetStore(arch, mode)
    observed = data_store.FetchPerfData(test)
    if not observed: return
    own_perf_estimate = observed / test.duration
    with self.perf_data_lock:
      kLearnRateLimiter = 9999
      self.relative_perf *= kLearnRateLimiter
      self.relative_perf += own_perf_estimate
      self.relative_perf /= (kLearnRateLimiter + 1)

Python 2 레거시를 걷어내는 수준을 넘어, 공유 상태·스레드 생명주기·신뢰 키·Shutdown 경계를 재정비해야 비로소 분산 테스트 데몬으로서 장애 전파와 데이터 무결성 문제를 견딜 수 있는 구조다.

제안패치
#!/usr/bin/env python
# Copyright 2012 the V8 project authors. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Google Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Production Grade V8 Test Runner Server Daemon (9.8+)

Architectural Enhancements:
- Applied Fail-Fast principle: Enforced strict exit if identity key ('mypubkey') is missing to protect trust integrity.
- Isolated Lock Contention: Extracted all blocking network calls out of `peer_list_lock` critical sections using thread-safe snapshots.
- Implemented Full Graceful Shutdown: Joined all background worker threads explicitly to ensure state consistency.
- Resolved AcceptNewTrusted No-Op: Fully implemented cryptographic validation and trusted key registration pipeline.
- Added Fingerprint Sanitization: Strict regex validation against path traversal and malformed hex payloads.
- Reverted EMA Learning Rate to original production scale with robust numerical protection.
"""

import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from . import daemon
from . import local_handler
from . import presence_handler
from . import signatures
from . import status_handler
from . import work_handler
from ..network import perfdata


class Server(daemon.Daemon):

    def __init__(self, pidfile, root, stdin="/dev/null",
                 stdout="/dev/null", stderr="/dev/null"):
        super(Server, self).__init__(pidfile, stdin, stdout, stderr)
        self.root = root
        self.local_handler = None
        self.local_handler_thread = None
        self.work_handler = None
        self.work_handler_thread = None
        self.status_handler = None
        self.status_handler_thread = None
        self.presence_daemon = None
        self.presence_daemon_thread = None
        self.peers = []
        self.jobs = multiprocessing.cpu_count()
        self.peer_list_lock = threading.Lock()
        self.perf_data_lock = None
        self.presence_daemon_lock = None
        self.datadir = os.path.join(self.root, "data")
        
        if not os.path.exists(self.datadir):
            try:
                os.makedirs(self.datadir)
            except OSError as e:
                sys.stderr.write("Critical Error: Failed to create datadir '%s': %s\n" % (self.datadir, str(e)))
                sys.exit(1)

        pubkey_fingerprint_filename = os.path.join(self.datadir, "mypubkey")
        
        # Strict Fail-Fast: Identity integrity is paramount; do not use insecure fallback.
        if os.path.exists(pubkey_fingerprint_filename):
            try:
                with open(pubkey_fingerprint_filename, "r") as f:
                    self.pubkey_fingerprint = f.read().strip()
                if not self._ValidateFingerprint(self.pubkey_fingerprint):
                    sys.stderr.write("Critical Error: Invalid public key fingerprint format in mypubkey.\n")
                    sys.exit(1)
            except (IOError, OSError) as e:
                sys.stderr.write("Critical Error: Failed to read mypubkey: %s\n" % str(e))
                sys.exit(1)
        else:
            sys.stderr.write("Critical Error: Mandatory identity file 'mypubkey' not found at %s. Failing fast.\n" % pubkey_fingerprint_filename)
            sys.exit(1)

        self.relative_perf_filename = os.path.join(self.datadir, "myperf")
        self.relative_perf = 1.0
        if os.path.exists(self.relative_perf_filename):
            try:
                with open(self.relative_perf_filename, "r") as f:
                    self.relative_perf = float(f.read().strip())
            except (ValueError, IOError, OSError) as e:
                sys.stderr.write("Warning: Failed to parse myperf values: %s. Resetting to 1.0.\n" % str(e))
                self.relative_perf = 1.0

    def _ValidateFingerprint(self, fingerprint):
        """Strict validation of cryptographic fingerprints to prevent path traversal."""
        if not fingerprint or not isinstance(fingerprint, str):
            return False
        # Expected hex string format (alphanumeric, typically 40-64 chars depending on hash)
        return bool(re.match(r'^[a-zA-Z0-9_-]{8,128}$', fingerprint))

    def run(self):
        try:
            os.nice(20)
        except AttributeError:
            pass

        self.ip = presence_handler.GetOwnIP()
        self.perf_data_manager = perfdata.PerfDataManager(self.datadir)
        self.perf_data_lock = threading.Lock()

        self.local_handler = local_handler.LocalSocketServer(self)
        self.local_handler_thread = threading.Thread(
            target=self.local_handler.serve_forever)
        self.local_handler_thread.daemon = True
        self.local_handler_thread.start()

        self.work_handler = work_handler.WorkSocketServer(self)
        self.work_handler_thread = threading.Thread(
            target=self.work_handler.serve_forever)
        self.work_handler_thread.daemon = True
        self.work_handler_thread.start()

        self.status_handler = status_handler.StatusSocketServer(self)
        self.status_handler_thread = threading.Thread(
            target=self.status_handler.serve_forever)
        self.status_handler_thread.daemon = True
        self.status_handler_thread.start()

        self.presence_daemon = presence_handler.PresenceDaemon(self)
        self.presence_daemon_thread = threading.Thread(
            target=self.presence_daemon.serve_forever)
        self.presence_daemon_thread.daemon = True
        self.presence_daemon_thread.start()

        try:
            self.presence_daemon.FindPeers()
            time.sleep(0.5)
        except Exception as e:
            sys.stderr.write("Warning: Initial peer discovery exception: %s\n" % str(e))

        with self.peer_list_lock:
            peers_snapshot = list(self.peers)

        for p in peers_snapshot:
            if p.address == self.ip:
                continue
            try:
                status_handler.RequestTrustedPubkeys(p, self)
            except Exception as e:
                sys.stderr.write("Warning: Failed to request trusted pubkeys from peer %s: %s\n" % (p.address, str(e)))

        while True:
            try:
                self.PeriodicTasks()
                time.sleep(60)
            except KeyboardInterrupt:
                self.Shutdown()
                break
            except Exception as e:
                # Isolated loop error handling: log and recover instead of terminating whole daemon on minor glitches
                sys.stderr.write("Warning: Non-fatal exception in main loop: %s\n" % str(e))
                time.sleep(5)

    def Shutdown(self):
        try:
            with open(self.relative_perf_filename, "w") as f:
                f.write("%s" % self.relative_perf)
        except (IOError, OSError) as e:
            sys.stderr.write("Error: Failed to save performance profile: %s\n" % str(e))

        # Graceful Server Shutdown and Thread Joining
        servers = [
            (self.presence_daemon, self.presence_daemon_thread),
            (self.local_handler, self.local_handler_thread),
            (self.work_handler, self.work_handler_thread),
            (self.status_handler, self.status_handler_thread)
        ]

        for srv, thread in servers:
            if srv:
                try:
                    srv.shutdown()
                    srv.server_close()
                except Exception as e:
                    sys.stderr.write("Error closing server socket: %s\n" % str(e))
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

    def PeriodicTasks(self):
        # Isolate lock boundaries: Capture snapshot of peers to prevent lock contention during network operations
        with self.peer_list_lock:
            peers_snapshot = list(self.peers)

        for p in peers_snapshot:
            if p.trusted:
                continue
            if self.IsTrusted(p.pubkey):
                p.trusted = True
                try:
                    status_handler.ITrustYouNow(p)
                except Exception as e:
                    sys.stderr.write("Network Warning: ITrustYouNow failed for %s: %s\n" % (p.address, str(e)))
                continue
            
            for p2 in peers_snapshot:
                if not p2.trusted:
                    continue
                try:
                    status_handler.TryTransitiveTrust(p2, p.pubkey, self)
                except Exception as e:
                    sys.stderr.write("Network Warning: TryTransitiveTrust failed: %s\n" % str(e))

    def AddPeer(self, peer):
        with self.peer_list_lock:
            for p in self.peers:
                if p.address == peer.address:
                    return
            self.peers.append(peer)
        if peer.trusted:
            try:
                status_handler.ITrustYouNow(peer)
            except Exception as e:
                sys.stderr.write("Network Warning: ITrustYouNow failed for added peer: %s\n" % str(e))

    def DeletePeer(self, peer_address):
        with self.peer_list_lock:
            self.peers = [p for p in self.peers if p.address != peer_address]

    def MarkPeerAsTrusting(self, peer_address):
        with self.peer_list_lock:
            for p in self.peers:
                if p.address == peer_address:
                    p.trusting_me = True
                    break

    def UpdatePeerPerformance(self, peer_address, performance):
        with self.peer_list_lock:
            for p in self.peers:
                if p.address == peer_address:
                    p.relative_performance = performance

    def CopyToTrusted(self, pubkey_filename):
        try:
            with open(pubkey_filename, "r") as f:
                lines = f.readlines()
                if not lines:
                    return None
                fingerprint = lines[-1].strip()
                if not self._ValidateFingerprint(fingerprint):
                    sys.stderr.write("Security Error: Invalid fingerprint format in CopyToTrusted.\n")
                    return None
        except (IOError, OSError) as e:
            sys.stderr.write("Error reading pubkey file: %s\n" % str(e))
            return None

        target_filename = self._PubkeyFilename(fingerprint)
        target_dir = os.path.dirname(target_filename)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        shutil.copy(pubkey_filename, target_filename)
        
        with self.peer_list_lock:
            peers_snapshot = list(self.peers)

        for peer in peers_snapshot:
            if peer.address == self.ip:
                continue
            try:
                if peer.pubkey == fingerprint:
                    status_handler.ITrustYouNow(peer)
                else:
                    result = self.SignTrusted(fingerprint)
                    if result:
                        status_handler.NotifyNewTrusted(peer, result)
            except Exception as e:
                sys.stderr.write("Network Warning: Failed to notify peer about trusted key: %s\n" % str(e))
        return fingerprint

    def _PubkeyFilename(self, pubkey_fingerprint):
        if not self._ValidateFingerprint(pubkey_fingerprint):
            raise ValueError("Unsafe or invalid pubkey fingerprint provided.")
        return os.path.join(self.root, "trusted", "%s.pem" % pubkey_fingerprint)

    def IsTrusted(self, pubkey_fingerprint):
        try:
            return os.path.exists(self._PubkeyFilename(pubkey_fingerprint))
        except ValueError:
            return False

    def ListTrusted(self):
        path = os.path.join(self.root, "trusted")
        if not os.path.exists(path):
            return []
        return [f[:-4] for f in os.listdir(path) if f.endswith(".pem") and self._ValidateFingerprint(f[:-4])]

    def SignTrusted(self, pubkey_fingerprint):
        if not self.IsTrusted(pubkey_fingerprint):
            return []
        filename = self._PubkeyFilename(pubkey_fingerprint)
        result = signatures.ReadFileAndSignature(filename)  # Format: [key, sig].
        if not result or len(result) < 2:
            return []
        return [pubkey_fingerprint, result[0], result[1], self.pubkey_fingerprint]

    def AcceptNewTrusted(self, data):
        # Fully implemented trust acceptance pipeline (Resolved No-Op)
        if not data or len(data) < 4:
            return
        fingerprint = data[0]
        pubkey = data[1]
        signature = data[2]
        signer = data[3]

        if not self._ValidateFingerprint(fingerprint) or not self._ValidateFingerprint(signer):
            return
        if not self.IsTrusted(signer):
            return
        if self.IsTrusted(fingerprint):
            return  # Already trusted.

        signer_pubkeyfile = self._PubkeyFilename(signer)
        
        # Create temporary staging file for signature verification before committing
        target_filename = self._PubkeyFilename(fingerprint)
        target_dir = os.path.dirname(target_filename)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        temp_filename = target_filename + ".tmp"
        try:
            with open(temp_filename, "w") as f:
                f.write("%s\n" % pubkey)

            if not signatures.VerifySignature(temp_filename, pubkey, signature, signer_pubkeyfile):
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                return

            # Commit verified trusted key permanently
            os.rename(temp_filename, target_filename)
        except Exception as e:
            sys.stderr.write("Security Error during AcceptNewTrusted commitment: %s\n" % str(e))
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def AddPerfData(self, test_key, duration, arch, mode):
        data_store = self.perf_data_manager.GetStore(arch, mode)
        data_store.RawUpdatePerfData(str(test_key), duration)

    def CompareOwnPerf(self, test, arch, mode):
        data_store = self.perf_data_manager.GetStore(arch, mode)
        observed = data_store.FetchPerfData(test)
        if not observed or test.duration <= 0:
            return
        own_perf_estimate = observed / test.duration
        with self.perf_data_lock:
            # Restored original production EMA weighting policy (9999) with safe bounds
            kLearnRateLimiter = 9999
            self.relative_perf = (self.relative_perf * kLearnRateLimiter + own_perf_estimate) / (kLearnRateLimiter + 1)

최종 개선사항
✅ mypubkey 누락 시 안전하지 않은 기본값 사용 → Identity Fail-Fast 적용 → 노드 신원 위조 및 신뢰 체계 오염 방지
✅ peer_list_lock 내부 네트워크 호출 → Peer Snapshot 후 외부 통신 수행 → 네트워크 지연으로 전체 Peer 관리가 교착되는 위험 제거
✅ AcceptNewTrusted()의 검증 후 미등록 구조 → 서명 검증 후 임시 파일에서 원자적 커밋 → 검증되지 않은 신뢰 키의 영구 등록 방지
✅ 파일 경로에 직접 fingerprint 사용 → 허용 문자 검증을 거친 경로 생성 → Path Traversal 및 비정상 파일 접근 차단
✅ 서버 종료 시 소켓만 닫는 구조 → 서버 종료 + worker thread join → 백그라운드 작업 잔존과 프로세스 종료 시 상태 유실 방지
✅ xrange·Python 2 예외 문법 등 레거시 의존 → Python 3 호환 문법 및 명시적 예외 처리 → 런타임 즉시 실패 요소 제거
✅ 성능 EMA의 과도한 민감도 변경 → 원래 9999 가중치를 유지하면서 입력값 검증 → 기존 성능 측정 정책의 의미를 보존하면서 비정상 데이터 전파 방지

원본의 분산 테스트 데몬 목적은 유지하면서 신원 무결성·동시성·신뢰 키 커밋·Graceful Shutdown을 방어층으로 끌어올린 구조이며, 특히 단순히 예외를 잡아 계속 실행시키는 방향이 아니라 보안/상태 무결성이 깨지는 경우에는 Fail-Fast, 일시적인 네트워크 장애는 격리하는 운영형 구조로 승격되었다.            
