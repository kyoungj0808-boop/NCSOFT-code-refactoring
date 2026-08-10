원본코드
# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import pdb
import sys


__all__ = ["set_trace"]


_stdin = [None]
_stdin_lock = multiprocessing.Lock()
try:
    _stdin_fd = sys.stdin.fileno()
except Exception:
    _stdin_fd = None


class MultiprocessingPdb(pdb.Pdb):
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage: `from fairseq import pdb; pdb.set_trace()`
    """

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        stdin_bak = sys.stdin
        with _stdin_lock:
            try:
                if _stdin_fd is not None:
                    if not _stdin[0]:
                        _stdin[0] = os.fdopen(_stdin_fd)
                    sys.stdin = _stdin[0]
                self.cmdloop()
            finally:
                sys.stdin = stdin_bak


def set_trace():
    pdb = MultiprocessingPdb()
    pdb.set_trace(sys._getframe().f_back)

멀티프로세싱 환경의 pdb 표준입력 충돌을 락과 nosigint로 효과적으로 제어한 실전형 디버깅 유틸리티지만, 모듈 로드 시점에 고정한 전역 FD와 공유 스트림 상태에 생명주기를 맡겨 프로세스 재생성·stdin 리다이렉션·FD 종료 같은 예외 상황에서는 디버깅 도구 자체가 장애 지점으로 뒤집힐 수 있는 레거시 구조다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import pdb
import sys


__all__ = ["set_trace"]


# 멀티프로세싱 환경 간 stdin 충돌 방지를 위한 유일한 전역 락
_stdin_lock = multiprocessing.Lock()


class MultiprocessingPdb(pdb.Pdb):
    """A Pdb wrapper that works safely in a multiprocessing environment,

    avoiding stale file descriptors and dynamic stdin re-routing issues.
    """

    def __init__(self):
        super().__init__(nosigint=True)

    def _cmdloop(self):
        stdin_bak = sys.stdin
        with _stdin_lock:
            try:
                # 디버거 진입 시점에 현재 프로세스의 표준 입력 FD를 동적으로 획득
                stdin_fd = None
                try:
                    stdin_fd = sys.stdin.fileno()
                except (AttributeError, io.UnsupportedOperation, OSError):
                    pass

                if stdin_fd is not None:
                    # 파일 디스크립터 직접 제어에 명확한 os.fdopen 사용 및 closefd=False로 소유권 보호
                    with os.fdopen(stdin_fd, "r", buffering=1, closefd=False) as f:
                        sys.stdin = f
                        self.cmdloop()
                else:
                    self.cmdloop()
            finally:
                sys.stdin = stdin_bak


def set_trace():
    debugger = MultiprocessingPdb()
    debugger.set_trace(sys._getframe().f_back)

최종 개선사항
✅ 모듈 로드 시점의 고정 stdin FD → 디버거 진입 시점 동적 FD 획득 → stdin 리다이렉션 및 프로세스 생명주기 변화 대응
✅ 전역 stdin 파일 객체 재사용 → os.fdopen(..., closefd=False) 기반 임시 스트림 사용 → 파일 객체 오염 및 FD 소유권 침해 방지
✅ 광범위한 예외 은닉 → AttributeError·io.UnsupportedOperation·OSError 중심의 명시적 방어 → 실제 장애 원인 은닉 최소화
✅ sys.stdin 변경 후 복구 보장 → try/finally 기반 원상복구 → 디버거 종료 후 프로세스 표준 입력 무결성 확보
✅ 멀티프로세스 stdin 접근 경쟁 → 단일 multiprocessing.Lock 직렬화 → 동시 디버깅 시 입력 스트림 충돌 방지
✅ 불필요한 전역 _stdin 캐시 제거 → 호출 단위 리소스 수명 관리 → stale state와 장기 실행 프로세스의 자원 오염 가능성 축소

원본의 멀티프로세싱 디버깅 목적은 유지하면서 stdin의 수명·FD 소유권·예외 경계를 재정립해, 레거시 응급처치 수준에서 프로세스 생명주기에 견디는 방어적 디버깅 유틸리티로 승격했다
