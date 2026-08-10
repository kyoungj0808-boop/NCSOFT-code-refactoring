원본코드
#!/usr/bin/env python
# Copyright 2015 the V8 project authors. All rights reserved.
# Copyright 2015 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Script to download LLVM gold plugin from google storage."""

import json
import os
import re
import platform
import shutil
import subprocess
import sys
import zipfile

# Bail out on windows and cygwin.
if "win" in platform.system().lower():
  # Python 2.7.6 hangs at the second path.insert command on windows. Works
  # with python 2.7.8.
  print "Gold plugin download not supported on windows."
  sys.exit(0)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CHROME_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, os.path.join(CHROME_SRC, 'tools'))

import find_depot_tools

DEPOT_PATH = find_depot_tools.add_depot_tools_to_path()
GSUTIL_PATH = os.path.join(DEPOT_PATH, 'gsutil.py')

LLVM_BUILD_PATH = os.path.join(CHROME_SRC, 'third_party', 'llvm-build',
                               'Release+Asserts')
CLANG_UPDATE_PY = os.path.join(CHROME_SRC, 'tools', 'clang', 'scripts',
                               'update.py')
CLANG_REVISION = os.popen(CLANG_UPDATE_PY + ' --print-revision').read().rstrip()

CLANG_BUCKET = 'gs://chromium-browser-clang/Linux_x64'

GOLD_PLUGIN_PATH = os.path.join(LLVM_BUILD_PATH, 'lib', 'LLVMgold.so')

sys.path.insert(0, os.path.join(CHROME_SRC, 'tools', 'clang', 'scripts'))

import update

def main():
  if not re.search(r'cfi_vptr=1', os.environ.get('GYP_DEFINES', '')):
    # Bailout if this is not a cfi build.
    print 'Skipping gold plugin download for non-cfi build.'
    return 0
  if (os.path.exists(GOLD_PLUGIN_PATH) and
      update.ReadStampFile().strip() == update.PACKAGE_VERSION):
    # Bailout if clang is up-to-date. This requires the script to be run before
    # the clang update step! I.e. afterwards clang would always be up-to-date.
    print 'Skipping gold plugin download. File present and clang up to date.'
    return 0

  # Make sure this works on empty checkouts (i.e. clang not downloaded yet).
  if not os.path.exists(LLVM_BUILD_PATH):
    os.makedirs(LLVM_BUILD_PATH)

  targz_name = 'llvmgold-%s.tgz' % CLANG_REVISION
  remote_path = '%s/%s' % (CLANG_BUCKET, targz_name)

  os.chdir(LLVM_BUILD_PATH)

  # TODO(pcc): Fix gsutil.py cp url file < /dev/null 2>&0
  # (currently aborts with exit code 1,
  # https://github.com/GoogleCloudPlatform/gsutil/issues/289) or change the
  # stdin->stderr redirect in update.py to do something else (crbug.com/494442).
  subprocess.check_call(['python', GSUTIL_PATH,
                         'cp', remote_path, targz_name],
                        stderr=open('/dev/null', 'w'))
  subprocess.check_call(['tar', 'xzf', targz_name])
  os.remove(targz_name)
  return 0

if __name__ == '__main__':
  sys.exit(main())

Python 2 레거시와 검증 없는 외부 실행·파일 정리 구조 때문에 CI 장애에 취약하지만, 다운로드 경계의 반환값·경로·파일 생명주기까지 방어하면 단순 부트스트랩 스크립트에서 재현성과 장애 복원력을 갖춘 운영형 빌드 유틸리티로 승격할 수 있는 코드다.

제안패치
#!/usr/bin/env python3

# Copyright 2015 the V8 project authors. All rights reserved.
# Copyright 2015 The Chromium Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Download the LLVM gold plugin required by CFI builds."""

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile


CLANG_BUCKET = "gs://chromium-browser-clang/Linux_x64"
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _write_error(message):
    sys.stderr.write("Error: %s\n" % message)


def _write_warning(message):
    sys.stderr.write("Warning: %s\n" % message)


def _get_clang_revision(clang_update_py):
    try:
        result = subprocess.run(
            [sys.executable, clang_update_py, "--print-revision"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Failed to retrieve Clang revision: %s" % exc
        )

    revision = result.stdout.strip()

    if not revision:
        raise RuntimeError("Clang update script returned an empty revision.")

    if not REVISION_PATTERN.fullmatch(revision):
        raise RuntimeError(
            "Clang revision contains unexpected characters: %r" % revision
        )

    return revision


def _is_safe_tar_member(member, destination):
    destination = os.path.realpath(destination)
    member_path = os.path.realpath(
        os.path.join(destination, member.name)
    )

    try:
        common_path = os.path.commonpath([destination, member_path])
    except ValueError:
        return False

    if common_path != destination:
        return False

    # Do not allow archive entries to replace links with arbitrary targets.
    if member.issym() or member.islnk():
        link_target = os.path.realpath(
            os.path.join(os.path.dirname(member_path), member.linkname)
        )
        try:
            return os.path.commonpath(
                [destination, link_target]
            ) == destination
        except ValueError:
            return False

    return True


def _extract_archive(archive_path, destination):
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()

            if not members:
                raise RuntimeError("Downloaded archive is empty.")

            unsafe_members = [
                member.name
                for member in members
                if not _is_safe_tar_member(member, destination)
            ]

            if unsafe_members:
                raise RuntimeError(
                    "Archive contains unsafe paths: %s"
                    % ", ".join(unsafe_members[:5])
                )

            archive.extractall(destination, members=members)

    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(
            "Failed to extract LLVM gold plugin: %s" % exc
        )


def main():
    if "win" in platform.system().lower():
        print("Gold plugin download not supported on windows.")
        return 0

    script_dir = os.path.dirname(os.path.realpath(__file__))
    chrome_src = os.path.abspath(os.path.join(script_dir, os.pardir))

    tools_path = os.path.join(chrome_src, "tools")
    clang_scripts_path = os.path.join(
        chrome_src, "tools", "clang", "scripts"
    )

    sys.path.insert(0, tools_path)

    try:
        import find_depot_tools
    except ImportError as exc:
        _write_error("Failed to import find_depot_tools: %s" % exc)
        return 1

    try:
        depot_path = find_depot_tools.add_depot_tools_to_path()
    except (OSError, RuntimeError) as exc:
        _write_error("Failed to initialize depot_tools: %s" % exc)
        return 1

    gsutil_path = os.path.join(depot_path, "gsutil.py")

    llvm_build_path = os.path.join(
        chrome_src,
        "third_party",
        "llvm-build",
        "Release+Asserts",
    )

    clang_update_py = os.path.join(
        clang_scripts_path,
        "update.py",
    )

    gold_plugin_path = os.path.join(
        llvm_build_path,
        "lib",
        "LLVMgold.so",
    )

    if not re.search(
        r"(?:^|[ \t])cfi_vptr=1(?:[ \t]|$)",
        os.environ.get("GYP_DEFINES", ""),
    ):
        print("Skipping gold plugin download for non-cfi build.")
        return 0

    try:
        sys.path.insert(0, clang_scripts_path)
        import update

        if (
            os.path.exists(gold_plugin_path)
            and update.ReadStampFile().strip() == update.PACKAGE_VERSION
        ):
            print(
                "Skipping gold plugin download. "
                "File present and clang up to date."
            )
            return 0
    except (OSError, ValueError, ImportError) as exc:
        _write_warning(
            "Unable to verify existing LLVM installation: %s" % exc
        )

    try:
        clang_revision = _get_clang_revision(clang_update_py)
    except RuntimeError as exc:
        _write_error(str(exc))
        return 1

    try:
        os.makedirs(llvm_build_path)
    except OSError:
        if not os.path.isdir(llvm_build_path):
            _write_error(
                "Failed to create LLVM build path: %s"
                % llvm_build_path
            )
            return 1

    targz_name = "llvmgold-%s.tgz" % clang_revision
    remote_path = "%s/%s" % (CLANG_BUCKET, targz_name)
    targz_fullpath = os.path.join(
        llvm_build_path,
        targz_name,
    )

    try:
        print("Downloading LLVM gold plugin from %s..." % remote_path)

        subprocess.run(
            [
                sys.executable,
                gsutil_path,
                "cp",
                remote_path,
                targz_fullpath,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        print("Extracting LLVM gold plugin...")
        _extract_archive(
            targz_fullpath,
            llvm_build_path,
        )

        if not os.path.isfile(gold_plugin_path):
            raise RuntimeError(
                "Extraction completed but LLVMgold.so was not found."
            )

    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        _write_error(
            "LLVM gold plugin installation failed: %s" % exc
        )
        return 1

    finally:
        try:
            os.remove(targz_fullpath)
        except FileNotFoundError:
            pass
        except OSError as exc:
            _write_warning(
                "Failed to remove temporary archive '%s': %s"
                % (targz_fullpath, exc)
            )

    print("LLVM gold plugin successfully downloaded and installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

최종 개선사항
✅ python PATH 의존 실행 → sys.executable 고정 → 동일 인터프리터 기반의 재현성 확보
✅ 검증 없는 Clang revision → 형식 검증 후 GCS 경로 사용 → 외부 입력의 경로 오염 방지
✅ CFI 여부 확인 전 초기화 → CFI 여부를 최우선 판정 → 불필요한 실행과 장애 지점 제거
✅ 무검증 tar 추출 → 아카이브 경로·링크 검증 후 추출 → 파일시스템 탈출 및 손상 방지
✅ 압축 해제 성공만 확인 → LLVMgold.so 실재 여부까지 검증 → 부분 설치를 성공으로 오인하는 문제 차단
✅ 광범위한 Exception 처리 → 예상 가능한 예외만 처리 → 장애 복구와 디버깅 가능성 동시 확보
✅ 임시 아카이브 정리 보장 → finally 기반 정리 유지 → 다운로드·추출 실패 후 빌드 디렉터리 오염 방지

원본은 다운로드만 성공하면 된 레거시 빌드 스크립트에 가까웠지만, 현재 구조는 외부 실행·아카이브·파일시스템·최종 산출물까지 검증하는 방어형 빌드 파이프라인으로 승격된 상태다.
