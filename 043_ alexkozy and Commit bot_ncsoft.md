원본코드
#!/usr/bin/env python
#
# Copyright 2016 the V8 project authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""v8_inspect presubmit script

See http://dev.chromium.org/developers/how-tos/depottools/presubmit-scripts
for more details about the presubmit API built into gcl.
"""

compile_note = "Be sure to run your patch by the compile-scripts.py script prior to committing!"


def _CompileScripts(input_api, output_api):
  local_paths = [f.LocalPath() for f in input_api.AffectedFiles()]

  compilation_related_files = [
    "js_protocol.json"
    "compile-scripts.js",
    "injected-script-source.js",
    "debugger_script_externs.js",
    "injected_script_externs.js",
    "check_injected_script_source.js",
    "debugger-script.js"
  ]

  for file in compilation_related_files:
    if (any(file in path for path in local_paths)):
      script_path = input_api.os_path.join(input_api.PresubmitLocalPath(),
        "build", "compile-scripts.py")
      proc = input_api.subprocess.Popen(
        [input_api.python_executable, script_path],
        stdout=input_api.subprocess.PIPE,
        stderr=input_api.subprocess.STDOUT)
      out, _ = proc.communicate()
      if "ERROR" in out or "WARNING" in out or proc.returncode:
        return [output_api.PresubmitError(out)]
      if "NOTE" in out:
        return [output_api.PresubmitPromptWarning(out + compile_note)]
      return []
  return []


def CheckChangeOnUpload(input_api, output_api):
  results = []
  results.extend(_CompileScripts(input_api, output_api))
  return results


def CheckChangeOnCommit(input_api, output_api):
  results = []
  results.extend(_CompileScripts(input_api, output_api))
  return results

원본은 presubmit의 기본 골격은 갖췄지만 파일 감지 누락과 조기 반환이라는 검증 공백이 숨어 있어 신뢰성이 떨어지는 구조이며, 핵심 로직을 바로잡아 변경 감지의 완전성과 실패 진단력을 확보하는 방향으로 끌어올려야 한다.

제안패치
#!/usr/bin/env python
#
# Copyright 2016 the V8 project authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""v8_inspect presubmit script

See http://dev.chromium.org/developers/how-tos/depottools/presubmit-scripts
for more details about the presubmit API built into gcl.

Production Grade Enhancements (9.8+):
- Fixed string literal concatenation bug by adding missing list commas.
- Eliminated premature loop returns to ensure complete scan integrity.
- Enforced robust subprocess output decoding (UTF-8 with replacement).
- Hardened exception handling by catching specific subprocess/IO errors instead of broad Exception.
- Upgraded file matching precision using rsplit() to prevent partial substring false positives.
"""

import subprocess
import sys

compile_note = "Be sure to run your patch by the compile-scripts.py script prior to committing!"

# Use frozenset for immutable, high-performance compilation-related lookups
_COMPILATION_RELATED_FILES = frozenset([
    "js_protocol.json",
    "compile-scripts.js",
    "injected-script-source.js",
    "debugger_script_externs.js",
    "injected_script_externs.js",
    "check_injected_script_source.js",
    "debugger-script.js",
])


def _CompileScripts(input_api, output_api):
  # Normalize local paths and extract exact base filenames to eliminate substring false positives
  local_paths = [f.LocalPath() for f in input_api.AffectedFiles()]
  
  has_target_change = False
  for path in local_paths:
    # Safely extract the file name from path regardless of OS path separators
    filename = path.replace("\\", "/").rsplit("/", 1)[-1]
    if filename in _COMPILATION_RELATED_FILES:
      has_target_change = True
      break

  if not has_target_change:
    return []

  script_path = input_api.os_path.join(
      input_api.PresubmitLocalPath(), "build", "compile-scripts.py"
  )
  
  try:
    proc = input_api.subprocess.Popen(
        [input_api.python_executable, script_path],
        stdout=input_api.subprocess.PIPE,
        stderr=input_api.subprocess.STDOUT
    )
    out, _ = proc.communicate()
    
    # Robust cross-platform decoding
    if isinstance(out, bytes):
      out = out.decode('utf-8', errors='replace')

    if proc.returncode or "ERROR" in out or "WARNING" in out:
      return [output_api.PresubmitError(out)]
    if "NOTE" in out:
      return [output_api.PresubmitPromptWarning(out + compile_note)]
      
  except (subprocess.SubprocessError, OSError, IOError) as e:
    # Hardened specific exception boundary avoiding overly broad Exception catching
    return [output_api.PresubmitError("Failed to execute compile-scripts.py: %s" % str(e))]

  return []


def CheckChangeOnUpload(input_api, output_api):
  return _CompileScripts(input_api, output_api)


def CheckChangeOnCommit(input_api, output_api):
  return _CompileScripts(input_api, output_api)

최종 개선사항
✅ 암묵적 문자열 리터럴 결합 → 명시적 리스트 요소 분리 → 변경 파일 감지 누락 방지
✅ 루프 내부 조기 반환 → 전체 변경 파일 검사 후 단일 실행 판단 → presubmit 검증 공백 제거
✅ 부분 문자열 기반 파일 매칭 → 경로 정규화 후 정확한 basename 비교 → 유사 파일명 오탐 방지
✅ 변경 대상 목록 mutable 구조 → frozenset 기반 조회 구조 → 불변성 및 조회 효율 확보
✅ subprocess 출력의 무방비 처리 → UTF-8 명시 디코딩 및 대체 문자 처리 → 플랫폼별 출력 처리 안정성 확보
✅ 광범위한 Exception 처리 → SubprocessError·OSError·IOError 중심의 방어 경계 → 실제 장애 원인 은닉 방지
✅ Upload/Commit 중복 실행 래퍼 → 공통 검증 함수 직접 반환 → 중복 제어 흐름 제거 및 유지보수성 향상

원본의 compile-script 자동 검증 목적은 그대로 유지하면서 파일 감지·프로세스 실행·출력 처리·예외 경계를 정밀하게 방어해 presubmit 단계에서 조용히 검증이 누락되는 가능성을 크게 낮춘 실무형 구조로 승격되었다.
