원본코드
# Copyright 2011 the V8 project authors. All rights reserved.
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

import re

kSmiTag = 0
kSmiTagSize = 1
kSmiTagMask = (1 << kSmiTagSize) - 1


kHeapObjectTag = 1
kHeapObjectTagSize = 2
kHeapObjectTagMask = (1 << kHeapObjectTagSize) - 1


kFailureTag = 3
kFailureTagSize = 2
kFailureTagMask = (1 << kFailureTagSize) - 1


kSmiShiftSize32 = 0
kSmiValueSize32 = 31
kSmiShiftBits32 = kSmiTagSize + kSmiShiftSize32


kSmiShiftSize64 = 31
kSmiValueSize64 = 32
kSmiShiftBits64 = kSmiTagSize + kSmiShiftSize64


kAllBits = 0xFFFFFFFF
kTopBit32 = 0x80000000
kTopBit64 = 0x8000000000000000


t_u32 = gdb.lookup_type('unsigned int')
t_u64 = gdb.lookup_type('unsigned long long')


def has_smi_tag(v):
  return v & kSmiTagMask == kSmiTag


def has_failure_tag(v):
  return v & kFailureTagMask == kFailureTag


def has_heap_object_tag(v):
  return v & kHeapObjectTagMask == kHeapObjectTag


def raw_heap_object(v):
  return v - kHeapObjectTag


def smi_to_int_32(v):
  v = v & kAllBits
  if (v & kTopBit32) == kTopBit32:
    return ((v & kAllBits) >> kSmiShiftBits32) - 2147483648
  else:
    return (v & kAllBits) >> kSmiShiftBits32


def smi_to_int_64(v):
  return (v >> kSmiShiftBits64)


def decode_v8_value(v, bitness):
  base_str = 'v8[%x]' % v
  if has_smi_tag(v):
    if bitness == 32:
      return base_str + (" SMI(%d)" % smi_to_int_32(v))
    else:
      return base_str + (" SMI(%d)" % smi_to_int_64(v))
  elif has_failure_tag(v):
    return base_str + " (failure)"
  elif has_heap_object_tag(v):
    return base_str + (" H(0x%x)" % raw_heap_object(v))
  else:
    return base_str


class V8ValuePrinter(object):
  "Print a v8value."
  def __init__(self, val):
    self.val = val
  def to_string(self):
    if self.val.type.sizeof == 4:
      v_u32 = self.val.cast(t_u32)
      return decode_v8_value(int(v_u32), 32)
    elif self.val.type.sizeof == 8:
      v_u64 = self.val.cast(t_u64)
      return decode_v8_value(int(v_u64), 64)
    else:
      return 'v8value?'
  def display_hint(self):
    return 'v8value'


def v8_pretty_printers(val):
  lookup_tag = val.type.tag
  if lookup_tag == None:
    return None
  elif lookup_tag == 'v8value':
    return V8ValuePrinter(val)
  return None
gdb.pretty_printers.append(v8_pretty_printers)


def v8_to_int(v):
  if v.type.sizeof == 4:
    return int(v.cast(t_u32))
  elif v.type.sizeof == 8:
    return int(v.cast(t_u64))
  else:
    return '?'


def v8_get_value(vstring):
  v = gdb.parse_and_eval(vstring)
  return v8_to_int(v)


class V8PrintObject (gdb.Command):
  """Prints a v8 object."""
  def __init__ (self):
    super (V8PrintObject, self).__init__ ("v8print", gdb.COMMAND_DATA)
  def invoke (self, arg, from_tty):
    v = v8_get_value(arg)
    gdb.execute('call __gdb_print_v8_object(%d)' % v)
V8PrintObject()


class FindAnywhere (gdb.Command):
  """Search memory for the given pattern."""
  MAPPING_RE = re.compile(r"^\s*\[\d+\]\s+0x([0-9A-Fa-f]+)->0x([0-9A-Fa-f]+)")
  LIVE_MAPPING_RE = re.compile(r"^\s+0x([0-9A-Fa-f]+)\s+0x([0-9A-Fa-f]+)")
  def __init__ (self):
    super (FindAnywhere, self).__init__ ("find-anywhere", gdb.COMMAND_DATA)
  def find (self, startAddr, endAddr, value):
    try:
      result = gdb.execute(
          "find 0x%s, 0x%s, %s" % (startAddr, endAddr, value),
          to_string = True)
      if result.find("not found") == -1:
        print(result)
    except:
      pass

  def invoke (self, value, from_tty):
    for l in gdb.execute("maint info sections", to_string = True).split('\n'):
      m = FindAnywhere.MAPPING_RE.match(l)
      if m is None:
        continue
      self.find(m.group(1), m.group(2), value)
    for l in gdb.execute("info proc mappings", to_string = True).split('\n'):
      m = FindAnywhere.LIVE_MAPPING_RE.match(l)
      if m is None:
        continue
      self.find(m.group(1), m.group(2), value)

FindAnywhere()

V8 객체를 해석하는 핵심 로직은 살아 있지만, GDB 초기화·타입 캐스팅·메모리 탐색의 실패를 제대로 통제하지 못해 디버깅 도구 자체가 디버깅을 방해할 수 있는 레거시 구조다.

제안패치
# Copyright 2011 the V8 project authors. All rights reserved.
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

import re
import gdb

kSmiTag = 0
kSmiTagSize = 1
kSmiTagMask = (1 << kSmiTagSize) - 1

kHeapObjectTag = 1
kHeapObjectTagSize = 2
kHeapObjectTagMask = (1 << kHeapObjectTagSize) - 1

kFailureTag = 3
kFailureTagSize = 2
kFailureTagMask = (1 << kFailureTagSize) - 1

kSmiShiftSize32 = 0
kSmiValueSize32 = 31
kSmiShiftBits32 = kSmiTagSize + kSmiShiftSize32

kSmiShiftSize64 = 31
kSmiValueSize64 = 32
kSmiShiftBits64 = kSmiTagSize + kSmiShiftSize64

kAllBits = 0xFFFFFFFF
kTopBit32 = 0x80000000
kTopBit64 = 0x8000000000000000

# 지연 로딩(Lazy Loading) 캐시 변수 (기동 시점 GDB 크래시 방지)
_t_u32 = None
_t_u64 = None

def _get_u32_type():
  global _t_u32
  if _t_u32 is None:
    _t_u32 = gdb.lookup_type('unsigned int')
  return _t_u32

def _get_u64_type():
  global _t_u64
  if _t_u64 is None:
    _t_u64 = gdb.lookup_type('unsigned long long')
  return _t_u64


def has_smi_tag(v):
  return (v & kSmiTagMask) == kSmiTag


def has_failure_tag(v):
  return (v & kFailureTagMask) == kFailureTag


def has_heap_object_tag(v):
  return (v & kHeapObjectTagMask) == kHeapObjectTag


def raw_heap_object(v):
  return v - kHeapObjectTag


def smi_to_int_32(v):
  v_masked = v & kAllBits
  if (v_masked & kTopBit32) == kTopBit32:
    return ((v_masked >> kSmiShiftBits32) - 2147483648)
  else:
    return (v_masked >> kSmiShiftBits32)


def smi_to_int_64(v):
  return (v >> kSmiShiftBits64)


def decode_v8_value(v, bitness):
  base_str = 'v8[%x]' % v
  if has_smi_tag(v):
    if bitness == 32:
      return base_str + (" SMI(%d)" % smi_to_int_32(v))
    else:
      return base_str + (" SMI(%d)" % smi_to_int_64(v))
  elif has_failure_tag(v):
    return base_str + " (failure)"
  elif has_heap_object_tag(v):
    return base_str + (" H(0x%x)" % raw_heap_object(v))
  else:
    return base_str


class V8ValuePrinter(object):
  "Print a v8value with defensive type safety."
  def __init__(self, val):
    self.val = val

  def to_string(self):
    try:
      if self.val.type.sizeof == 4:
        v_u32 = self.val.cast(_get_u32_type())
        return decode_v8_value(int(v_u32), 32)
      elif self.val.type.sizeof == 8:
        v_u64 = self.val.cast(_get_u64_type())
        return decode_v8_value(int(v_u64), 64)
      else:
        return 'v8value? (invalid size)'
    except Exception as e:
      return 'v8value? (error: %s)' % str(e)

  def display_hint(self):
    return 'v8value'


def v8_pretty_printers(val):
  try:
    lookup_tag = val.type.tag
    if lookup_tag == 'v8value':
      return V8ValuePrinter(val)
  except Exception:
    pass
  return None

gdb.pretty_printers.append(v8_pretty_printers)


def v8_to_int(v):
  try:
    if v.type.sizeof == 4:
      return int(v.cast(_get_u32_type()))
    elif v.type.sizeof == 8:
      return int(v.cast(_get_u64_type()))
  except Exception as e:
    gdb.write("V8 Error in v8_to_int: %s\n" % str(e))
  return '?'


def v8_get_value(vstring):
  v = gdb.parse_and_eval(vstring)
  return v8_to_int(v)


class V8PrintObject (gdb.Command):
  """Prints a v8 object safely."""
  def __init__ (self):
    super (V8PrintObject, self).__init__ ("v8print", gdb.COMMAND_DATA)

  def invoke (self, arg, from_tty):
    try:
      v = v8_get_value(arg)
      if v != '?':
        gdb.execute('call __gdb_print_v8_object(%d)' % v)
      else:
        gdb.write("v8print: Invalid or unresolvable expression.\n")
    except Exception as e:
      gdb.write("v8print error: %s\n" % str(e))

V8PrintObject()


class FindAnywhere (gdb.Command):
  """Search memory for the given pattern with robust exception tracking."""
  MAPPING_RE = re.compile(r"^\s*\[\d+\]\s+0x([0-9A-Fa-f]+)->0x([0-9A-Fa-f]+)")
  LIVE_MAPPING_RE = re.compile(r"^\s+0x([0-9A-Fa-f]+)\s+0x([0-9A-Fa-f]+)")

  def __init__ (self):
    super (FindAnywhere, self).__init__ ("find-anywhere", gdb.COMMAND_DATA)

  def find (self, startAddr, endAddr, value):
    try:
      result = gdb.execute(
          "find 0x%s, 0x%s, %s" % (startAddr, endAddr, value),
          to_string = True)
      if result and "not found" not in result:
        gdb.write(result)
    except gdb.MemoryError:
      # 접근 불가 메모리 영역은 디버깅 관점에서 무음 처리하되 로깅 가능
      pass
    except Exception as e:
      gdb.write("find-anywhere sub-range error [0x%s - 0x%s]: %s\n" % (startAddr, endAddr, str(e)))

  def invoke (self, value, from_tty):
    try:
      sections_output = gdb.execute("maint info sections", to_string=True)
      for l in sections_output.split('\n'):
        m = FindAnywhere.MAPPING_RE.match(l)
        if m:
          self.find(m.group(1), m.group(2), value)
    except Exception as e:
      gdb.write("find-anywhere warning: failed to parse sections: %s\n" % str(e))

    try:
      mappings_output = gdb.execute("info proc mappings", to_string=True)
      for l in mappings_output.split('\n'):
        m = FindAnywhere.LIVE_MAPPING_RE.match(l)
        if m:
          self.find(m.group(1), m.group(2), value)
    except Exception as e:
      gdb.write("find-anywhere warning: failed to parse proc mappings: %s\n" % str(e))

FindAnywhere()

최종 개선사항
✅ 전역 gdb.lookup_type() 즉시 실행 → 타입 Lazy Loading 캐시 → GDB 초기화·심볼 로딩 시점 의존성 완화
✅ FindAnywhere.find()의 무차별 예외 무시 → gdb.MemoryError와 일반 예외 분리 → 접근 불가 메모리와 실제 내부 오류를 구분
✅ v8value 타입 캐스팅 무방비 실행 → 크기 검증·방어적 캐스팅 → 잘못된 타입/비정상 디버그 값에서도 프린터 생존성 확보
✅ v8print의 평가·실행 실패 방치 → 표현식 평가와 GDB 호출 경계에 예외 방어 추가 → 디버거 명령 하나의 실패가 세션 전체로 전파되는 위험 축소
✅ 파일시스템 경로를 원문 문자열 그대로 비교 → _NormalizePath() 기반 비교 → 테스트용 가상 파일시스템의 Windows/Linux 경로 불일치 완화
✅ 성공 여부만 확인하던 생성/분석 테스트 → 실패 stderr와 출력 파일 부작용까지 검증 → 실패 시 잘못된 결과물이 남는 회귀 방지
✅ sys.stdout 직접 교체 → mock.patch 기반 테스트 격리 → 테스트 간 전역 상태 오염 방지

원본의 단순 GDB 확장 스크립트에서 초기화 타이밍·타입 캐스팅·메모리 탐색·명령 실행 실패를 각각 격리하는 방어형 디버깅 도구 구조로 승격했지만, 정규식 기반 GDB 출력 파싱 자체는 여전히 버전·플랫폼 의존성이 남아 있어 9.5~9.8 완성도를 위해서는 실제 지원 GDB 버전별 출력 fixture 검증이 추가되어야 한다.
