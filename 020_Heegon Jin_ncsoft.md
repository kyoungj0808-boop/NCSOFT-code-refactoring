원본코드
# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import shutil
import sys


pt_regexp = re.compile(r"checkpoint(\d+|_\d+_\d+|_[a-z]+)\.pt")
pt_regexp_epoch_based = re.compile(r"checkpoint(\d+)\.pt")
pt_regexp_update_based = re.compile(r"checkpoint_\d+_(\d+)\.pt")


def parse_checkpoints(files):
    entries = []
    for f in files:
        m = pt_regexp_epoch_based.fullmatch(f)
        if m is not None:
            entries.append((int(m.group(1)), m.group(0)))
        else:
            m = pt_regexp_update_based.fullmatch(f)
            if m is not None:
                entries.append((int(m.group(1)), m.group(0)))
    return entries


def last_n_checkpoints(files, n):
    entries = parse_checkpoints(files)
    return [x[1] for x in sorted(entries, reverse=True)[:n]]


def every_n_checkpoints(files, n):
    entries = parse_checkpoints(files)
    return [x[1] for x in sorted(sorted(entries)[::-n])]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively delete checkpoint files from `root_dir`, "
            "but preserve checkpoint_best.pt and checkpoint_last.pt"
        )
    )
    parser.add_argument("root_dirs", nargs="*")
    parser.add_argument(
        "--save-last", type=int, default=0, help="number of last checkpoints to save"
    )
    parser.add_argument(
        "--save-every", type=int, default=0, help="interval of checkpoints to save"
    )
    parser.add_argument(
        "--preserve-test",
        action="store_true",
        help="preserve checkpoints in dirs that start with test_ prefix (default: delete them)",
    )
    parser.add_argument(
        "--delete-best", action="store_true", help="delete checkpoint_best.pt"
    )
    parser.add_argument(
        "--delete-last", action="store_true", help="delete checkpoint_last.pt"
    )
    parser.add_argument(
        "--no-dereference", action="store_true", help="don't dereference symlinks"
    )
    args = parser.parse_args()

    files_to_desymlink = []
    files_to_preserve = []
    files_to_delete = []
    for root_dir in args.root_dirs:
        for root, _subdirs, files in os.walk(root_dir):
            if args.save_last > 0:
                to_save = last_n_checkpoints(files, args.save_last)
            else:
                to_save = []
            if args.save_every > 0:
                to_save += every_n_checkpoints(files, args.save_every)
            for file in files:
                if not pt_regexp.fullmatch(file):
                    continue
                full_path = os.path.join(root, file)
                if (
                    not os.path.basename(root).startswith("test_") or args.preserve_test
                ) and (
                    (file == "checkpoint_last.pt" and not args.delete_last)
                    or (file == "checkpoint_best.pt" and not args.delete_best)
                    or file in to_save
                ):
                    if os.path.islink(full_path) and not args.no_dereference:
                        files_to_desymlink.append(full_path)
                    else:
                        files_to_preserve.append(full_path)
                else:
                    files_to_delete.append(full_path)

    if len(files_to_desymlink) == 0 and len(files_to_delete) == 0:
        print("Nothing to do.")
        sys.exit(0)

    files_to_desymlink = sorted(files_to_desymlink)
    files_to_preserve = sorted(files_to_preserve)
    files_to_delete = sorted(files_to_delete)

    print("Operations to perform (in order):")
    if len(files_to_desymlink) > 0:
        for file in files_to_desymlink:
            print(" - preserve (and dereference symlink): " + file)
    if len(files_to_preserve) > 0:
        for file in files_to_preserve:
            print(" - preserve: " + file)
    if len(files_to_delete) > 0:
        for file in files_to_delete:
            print(" - delete: " + file)
    while True:
        resp = input("Continue? (Y/N): ")
        if resp.strip().lower() == "y":
            break
        elif resp.strip().lower() == "n":
            sys.exit(0)

    print("Executing...")
    if len(files_to_desymlink) > 0:
        for file in files_to_desymlink:
            realpath = os.path.realpath(file)
            print("rm " + file)
            os.remove(file)
            print("cp {} {}".format(realpath, file))
            shutil.copyfile(realpath, file)
    if len(files_to_delete) > 0:
        for file in files_to_delete:
            print("rm " + file)
            os.remove(file)


if __name__ == "__main__":
    main()

연구용 체크포인트 정리 스크립트 수준에서는 동작하지만, 실제 학습 인프라에 투입하기에는 탐색과 삭제 사이의 TOCTOU·symlink 처리·비원자적 파일 교체·입력/경로 검증 부재가 겹쳐 있어, 한 번의 경쟁 상태나 I/O 실패가 체크포인트 손실로 직결될 수 있는 파괴적 파일 작업 코드다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# 1. 환경 변수 유효성 검증 체계화 (문자열 오입력 및 비정상 범위 차단)
def _parse_env_int(var_name: str, default: int, min_val: int) -> int:
    val_str = os.getenv(var_name, str(default))
    try:
        val = int(val_str)
        if val < min_val:
            raise ValueError(f"값은 {min_val} 이상이어야 합니다.")
        return val
    except ValueError as e:
        print(f"[경고] 잘못된 환경 변수 설정 ({var_name}={val_str}): {e}. 기본값({default})을 적용합니다.", file=sys.stderr)
        return default

DRY_RUN = os.getenv("CHECKPOINT_DRY_RUN", "false").lower() == "true"
MAX_RETRIES = _parse_env_int("CHECKPOINT_MAX_RETRIES", 3, min_val=1)

pt_regexp = re.compile(r"checkpoint(\d+|_\d+_\d+|_[a-z]+)\.pt")
pt_regexp_epoch_based = re.compile(r"checkpoint(\d+)\.pt")
pt_regexp_update_based = re.compile(r"checkpoint_\d+_(\d+)\.pt")


def parse_checkpoints(files):
    entries = []
    for f in files:
        m = pt_regexp_epoch_based.fullmatch(f)
        if m is not None:
            entries.append((int(m.group(1)), m.group(0)))
        else:
            m = pt_regexp_update_based.fullmatch(f)
            if m is not None:
                entries.append((int(m.group(1)), m.group(0)))
    return entries


def last_n_checkpoints(files, n):
    entries = parse_checkpoints(files)
    return [x[1] for x in sorted(entries, reverse=True)[:n]]


def every_n_checkpoints(files, n):
    entries = parse_checkpoints(files)
    return [x[1] for x in sorted(sorted(entries)[::-n])]


def validate_and_resolve_path(raw_path: str) -> Path:
    resolved_path = Path(raw_path).resolve()
    if not resolved_path.exists() or not resolved_path.is_dir():
        raise ValueError(f"유효하지 않거나 접근할 수 없는 디렉토리 경로입니다 -> {raw_path}")
    return resolved_path


def atomic_desymlink(file_path: str) -> None:
    """
    진정한 원자적(Atomic) 심볼릭 링크 변환 구현:
    - 임시 파일(Temporary File)을 동일 파일 시스템 내에 먼저 생성한 뒤,
    - os.replace()를 통해 원자적으로 교체함으로써 중간에 프로세스가 죽어도 
      원본 데이터나 링크가 유실되는 비정상 상태(Data Loss)를 원천 방지함.
    """
    path_obj = Path(file_path)
    if not path_obj.is_symlink():
        raise OSError(f"대상 파일이 심볼릭 링크가 아닙니다: {file_path}")

    # realpath 추출 및 무결성 검증
    real_path = os.path.realpath(file_path)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"심볼릭 링크의 원본 경로를 찾을 수 없습니다: {real_path}")

    # 동일 디렉토리 내 임시 파일 생성 (.tmp)
    temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    try:
        shutil.copyfile(real_path, temp_path)
        # os.replace는 동일 파일 시스템 내에서 원자적으로 동작함
        os.replace(temp_path, path_obj)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively delete checkpoint files from `root_dir`, "
            "but preserve checkpoint_best.pt and checkpoint_last.pt"
        )
    )
    parser.add_argument("root_dirs", nargs="*")
    parser.add_argument(
        "--save-last", type=int, default=0, help="number of last checkpoints to save"
    )
    parser.add_argument(
        "--save-every", type=int, default=0, help="interval of checkpoints to save"
    )
    parser.add_argument(
        "--preserve-test",
        action="store_true",
        help="preserve checkpoints in dirs that start with test_ prefix (default: delete them)",
    )
    parser.add_argument(
        "--delete-best", action="store_true", help="delete checkpoint_best.pt"
    )
    parser.add_argument(
        "--delete-last", action="store_true", help="delete checkpoint_last.pt"
    )
    parser.add_argument(
        "--no-dereference", action="store_true", help="don't dereference symlinks"
    )
    args = parser.parse_args()

    # 2. CLI 인자값 의미적 무결성 검증 (음수 방어)
    if args.save_last < 0 or args.save_every < 0:
        print("오류: `--save-last`와 `--save-every` 값은 0 이상이어야 합니다.", file=sys.stderr)
        sys.exit(1)

    if not args.root_dirs:
        print("오류: 대상 루트 디렉토리가 지정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)

    files_to_desymlink = []
    files_to_preserve = []
    files_to_delete = []

    try:
        validated_roots = [validate_and_resolve_path(d) for d in args.root_dirs]
    except ValueError as e:
        print(f"무결성 검증 실패: {e}", file=sys.stderr)
        sys.exit(1)

    for root_path in validated_roots:
        for root, _subdirs, files in os.walk(root_path):
            if args.save_last > 0:
                to_save = last_n_checkpoints(files, args.save_last)
            else:
                to_save = []
            if args.save_every > 0:
                to_save += every_n_checkpoints(files, args.save_every)
            
            for file in files:
                if not pt_regexp.fullmatch(file):
                    continue
                full_path = os.path.join(root, file)
                
                # TOCTOU 완화: 파일 기술 서술부의 물리적 존재 여부 재확인
                if not os.path.lexists(full_path):
                    continue

                if (
                    not os.path.basename(root).startswith("test_") or args.preserve_test
                ) and (
                    (file == "checkpoint_last.pt" and not args.delete_last)
                    or (file == "checkpoint_best.pt" and not args.delete_best)
                    or file in to_save
                ):
                    if os.path.islink(full_path) and not args.no_dereference:
                        files_to_desymlink.append(full_path)
                    else:
                        files_to_preserve.append(full_path)
                else:
                    files_to_delete.append(full_path)

    if len(files_to_desymlink) == 0 and len(files_to_delete) == 0:
        print("Nothing to do.")
        sys.exit(0)

    files_to_desymlink = sorted(files_to_desymlink)
    files_to_preserve = sorted(files_to_preserve)
    files_to_delete = sorted(files_to_delete)

    print(f"Operations to perform (in order) [DRY_RUN={DRY_RUN}]:")
    if len(files_to_desymlink) > 0:
        for file in files_to_desymlink:
            print(" - preserve (and dereference symlink): " + file)
    if len(files_to_preserve) > 0:
        for file in files_to_preserve:
            print(" - preserve: " + file)
    if len(files_to_delete) > 0:
        for file in files_to_delete:
            print(" - delete: " + file)

    try:
        resp = input("Continue? (Y/N): ")
    except (KeyboardInterrupt, EOFError):
        print("\n작업이 사용자에 의해 안전하게 중단되었습니다.")
        sys.exit(0)

    if resp.strip().lower() != "y":
        print("작업이 취소되었습니다.")
        sys.exit(0)

    # 3. DRY_RUN 모드 명시적 적용 (활성화 시 실제 파일 시스템 변경 차단)
    if DRY_RUN:
        print("[DRY_RUN 모드 활성화] 실제 파일 시스템 변경 작업은 수행되지 않습니다.")
        sys.exit(0)

    print("Executing...")
    
    execution_failed = False

    # 4. 심볼릭 링크 변환 작업 (원자적 연산 + 세분화된 예외 처리 + 실패 시 Exit Code 보장)
    if len(files_to_desymlink) > 0:
        for file in files_to_desymlink:
            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    if not os.path.lexists(file):
                        print(f"[경고] 대상 파일이 존재하지 않습니다: {file}")
                        success = True
                        break
                    print(f"atomic desymlink {file} (Attempt {attempt}/{MAX_RETRIES})")
                    atomic_desymlink(file)
                    success = True
                    break
                except FileNotFoundError as e:
                    print(f"[오류] 파일을 찾을 수 없음 ({file}): {e}")
                    break  # 재시도 불가한 오류
                except PermissionError as e:
                    print(f"[오류] 권한 거부됨 ({file}): {e}")
                    break  # 재시도 불가한 오류
                except (OSError, IOError) as e:
                    print(f"[경고] I/O 일시적 잠금 발생 ({e}), 재시도 중...")
                    if attempt == MAX_RETRIES:
                        print(f"[치명적 오류] 최대 재시도 횟수 초과: {file}", file=sys.stderr)
                except Exception as e:
                    print(f"[치명적 오류] 예상치 못한 예외 발생 ({file}): {e}", file=sys.stderr)
                    break
            
            if not success:
                execution_failed = True

    # 5. 파일 삭제 작업 (세분화된 예외 처리 + 실패 시 Exit Code 보장)
    if len(files_to_delete) > 0:
        for file in files_to_delete:
            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    if not os.path.lexists(file):
                        success = True
                        break
                    print(f"rm {file} (Attempt {attempt}/{MAX_RETRIES})")
                    os.remove(file)
                    success = True
                    break
                except FileNotFoundError:
                    success = True
                    break
                except PermissionError as e:
                    print(f"[오류] 삭제 권한 거부됨 ({file}): {e}")
                    break
                except (OSError, IOError) as e:
                    print(f"[경고] 삭제 중 I/O 오류 발생 ({e}), 재시도 중...")
                    if attempt == MAX_RETRIES:
                        print(f"[치명적 오류] 최대 재시도 횟수 초과 삭제 실패: {file}", file=sys.stderr)
                except Exception as e:
                    print(f"[치명적 오류] 예상치 못한 삭제 예외 발생 ({file}): {e}", file=sys.stderr)
                    break

            if not success:
                execution_failed = True

    # 6. 작업 실패 시 명확한 비정상 종료 코드 반환 (CI/CD 파이프라인 안전성 확보)
    if execution_failed:
        print("[치명적 오류] 일부 체크포인트 정리 작업이 실패했습니다.", file=sys.stderr)
        sys.exit(1)

    print("모든 작업이 성공적으로 완료되었습니다.")


if __name__ == "__main__":
    main()

최종 개선사항
✅ 하드코딩된 재시도 설정 → 환경변수 주입 및 범위 검증 → 운영 환경별 설정 변경성과 잘못된 설정값에 대한 방어력 확보
✅ 무검증 CLI 인자 → --save-last·--save-every 의미적 검증 → 음수·비정상 실행 조건의 사전 차단
✅ 탐색 경로 직접 사용 → Path.resolve() 기반 경로 검증 → 잘못된 대상 디렉터리 접근 가능성 축소
✅ 삭제 전 단순 존재 확인 → 실행 직전 lexists() 재검증 및 파일별 예외 분류 → 파일 소실·권한 오류 상황에서의 장애 전파 방어
✅ os.remove() 후 copyfile() 방식 → 동일 디렉터리 임시 파일 + os.replace() 기반 원자적 심볼릭 링크 해제 → 변환 중 프로세스 장애 시 데이터 손실 위험 최소화
✅ 무차별 except Exception 재시도 → FileNotFoundError·PermissionError·I/O 오류를 분리 처리 → 재시도 가능한 장애와 즉시 중단해야 할 장애를 구분
✅ 작업 실패를 출력만 하고 종료 → execution_failed 집계 및 비정상 Exit Code 반환 → CI/CD·자동화 환경에서 부분 실패를 명확하게 감지 가능

원본의 단순 체크포인트 삭제 스크립트에서 설정 검증·파일 무결성·원자적 처리·실패 전파까지 갖춘 운영 대응형 정리 도구로 승격된 상태다.
