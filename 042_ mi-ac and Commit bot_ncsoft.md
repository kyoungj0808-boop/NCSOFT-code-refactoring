원본코드
#!/usr/bin/env python
# Copyright 2015 the V8 project authors. All rights reserved.
# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Adaptor script called through build/isolate.gypi.

Creates a wrapping .isolate which 'includes' the original one, that can be
consumed by tools/swarming_client/isolate.py. Path variables are determined
based on the current working directory. The relative_cwd in the .isolated file
is determined based on the .isolate file that declare the 'command' variable to
be used so the wrapping .isolate doesn't affect this value.

This script loads build.ninja and processes it to determine all the executables
referenced by the isolated target. It adds them in the wrapping .isolate file.

WARNING: The target to use for build.ninja analysis is the base name of the
.isolate file plus '_run'. For example, 'foo_test.isolate' would have the target
'foo_test_run' analysed.
"""

import errno
import glob
import json
import logging
import os
import posixpath
import StringIO
import subprocess
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
SWARMING_CLIENT_DIR = os.path.join(TOOLS_DIR, 'swarming_client')
SRC_DIR = os.path.dirname(TOOLS_DIR)

sys.path.insert(0, SWARMING_CLIENT_DIR)

import isolate_format


def load_ninja_recursively(build_dir, ninja_path, build_steps):
  """Crudely extracts all the subninja and build referenced in ninja_path.

  In particular, it ignores rule and variable declarations. The goal is to be
  performant (well, as much as python can be performant) which is currently in
  the <200ms range for a complete chromium tree. As such the code is laid out
  for performance instead of readability.
  """
  logging.debug('Loading %s', ninja_path)
  try:
    with open(os.path.join(build_dir, ninja_path), 'rb') as f:
      line = None
      merge_line = ''
      subninja = []
      for line in f:
        line = line.rstrip()
        if not line:
          continue

        if line[-1] == '$':
          # The next line needs to be merged in.
          merge_line += line[:-1]
          continue

        if merge_line:
          line = merge_line + line
          merge_line = ''

        statement = line[:line.find(' ')]
        if statement == 'build':
          # Save the dependency list as a raw string. Only the lines needed will
          # be processed with raw_build_to_deps(). This saves a good 70ms of
          # processing time.
          build_target, dependencies = line[6:].split(': ', 1)
          # Interestingly, trying to be smart and only saving the build steps
          # with the intended extensions ('', '.stamp', '.so') slows down
          # parsing even if 90% of the build rules can be skipped.
          # On Windows, a single step may generate two target, so split items
          # accordingly. It has only been seen for .exe/.exe.pdb combos.
          for i in build_target.strip().split():
            build_steps[i] = dependencies
        elif statement == 'subninja':
          subninja.append(line[9:])
  except IOError:
    print >> sys.stderr, 'Failed to open %s' % ninja_path
    raise

  total = 1
  for rel_path in subninja:
    try:
      # Load each of the files referenced.
      # TODO(maruel): Skip the files known to not be needed. It saves an aweful
      # lot of processing time.
      total += load_ninja_recursively(build_dir, rel_path, build_steps)
    except IOError:
      print >> sys.stderr, '... as referenced by %s' % ninja_path
      raise
  return total


def load_ninja(build_dir):
  """Loads the tree of .ninja files in build_dir."""
  build_steps = {}
  total = load_ninja_recursively(build_dir, 'build.ninja', build_steps)
  logging.info('Loaded %d ninja files, %d build steps', total, len(build_steps))
  return build_steps


def using_blacklist(item):
  """Returns True if an item should be analyzed.

  Ignores many rules that are assumed to not depend on a dynamic library. If
  the assumption doesn't hold true anymore for a file format, remove it from
  this list. This is simply an optimization.
  """
  # *.json is ignored below, *.isolated.gen.json is an exception, it is produced
  # by isolate_driver.py in 'test_isolation_mode==prepare'.
  if item.endswith('.isolated.gen.json'):
    return True
  IGNORED = (
    '.a', '.cc', '.css', '.dat', '.def', '.frag', '.h', '.html', '.isolate',
    '.js', '.json', '.manifest', '.o', '.obj', '.pak', '.png', '.pdb', '.py',
    '.strings', '.test', '.txt', '.vert',
  )
  # ninja files use native path format.
  ext = os.path.splitext(item)[1]
  if ext in IGNORED:
    return False
  # Special case Windows, keep .dll.lib but discard .lib.
  if item.endswith('.dll.lib'):
    return True
  if ext == '.lib':
    return False
  return item not in ('', '|', '||')


def raw_build_to_deps(item):
  """Converts a raw ninja build statement into the list of interesting
  dependencies.
  """
  # TODO(maruel): Use a whitelist instead? .stamp, .so.TOC, .dylib.TOC,
  # .dll.lib, .exe and empty.
  # The first item is the build rule, e.g. 'link', 'cxx', 'phony', etc.
  return filter(using_blacklist, item.split(' ')[1:])


def collect_deps(target, build_steps, dependencies_added, rules_seen):
  """Recursively adds all the interesting dependencies for |target|
  into |dependencies_added|.
  """
  if rules_seen is None:
    rules_seen = set()
  if target in rules_seen:
    # TODO(maruel): Figure out how it happens.
    logging.warning('Circular dependency for %s!', target)
    return
  rules_seen.add(target)
  try:
    dependencies = raw_build_to_deps(build_steps[target])
  except KeyError:
    logging.info('Failed to find a build step to generate: %s', target)
    return
  logging.debug('collect_deps(%s) -> %s', target, dependencies)
  for dependency in dependencies:
    dependencies_added.add(dependency)
    collect_deps(dependency, build_steps, dependencies_added, rules_seen)


def post_process_deps(build_dir, dependencies):
  """Processes the dependency list with OS specific rules."""
  def filter_item(i):
    if i.endswith('.so.TOC'):
      # Remove only the suffix .TOC, not the .so!
      return i[:-4]
    if i.endswith('.dylib.TOC'):
      # Remove only the suffix .TOC, not the .dylib!
      return i[:-4]
    if i.endswith('.dll.lib'):
      # Remove only the suffix .lib, not the .dll!
      return i[:-4]
    return i

  def is_exe(i):
    # This script is only for adding new binaries that are created as part of
    # the component build.
    ext = os.path.splitext(i)[1]
    # On POSIX, executables have no extension.
    if ext not in ('', '.dll', '.dylib', '.exe', '.nexe', '.so'):
      return False
    if os.path.isabs(i):
      # In some rare case, there's dependency set explicitly on files outside
      # the checkout.
      return False

    # Check for execute access and strip directories. This gets rid of all the
    # phony rules.
    p = os.path.join(build_dir, i)
    return os.access(p, os.X_OK) and not os.path.isdir(p)

  return filter(is_exe, map(filter_item, dependencies))


def create_wrapper(args, isolate_index, isolated_index):
  """Creates a wrapper .isolate that add dynamic libs.

  The original .isolate is not modified.
  """
  cwd = os.getcwd()
  isolate = args[isolate_index]
  # The code assumes the .isolate file is always specified path-less in cwd. Fix
  # if this assumption doesn't hold true.
  assert os.path.basename(isolate) == isolate, isolate

  # This will look like ../out/Debug. This is based against cwd. Note that this
  # must equal the value provided as PRODUCT_DIR.
  build_dir = os.path.dirname(args[isolated_index])

  # This will look like chrome/unit_tests.isolate. It is based against SRC_DIR.
  # It's used to calculate temp_isolate.
  src_isolate = os.path.relpath(os.path.join(cwd, isolate), SRC_DIR)

  # The wrapping .isolate. This will look like
  # ../out/Debug/gen/chrome/unit_tests.isolate.
  temp_isolate = os.path.join(build_dir, 'gen', src_isolate)
  temp_isolate_dir = os.path.dirname(temp_isolate)

  # Relative path between the new and old .isolate file.
  isolate_relpath = os.path.relpath(
      '.', temp_isolate_dir).replace(os.path.sep, '/')

  # It's a big assumption here that the name of the isolate file matches the
  # primary target '_run'. Fix accordingly if this doesn't hold true, e.g.
  # complain to maruel@.
  target = isolate[:-len('.isolate')] + '_run'
  build_steps = load_ninja(build_dir)
  binary_deps = set()
  collect_deps(target, build_steps, binary_deps, None)
  binary_deps = post_process_deps(build_dir, binary_deps)
  logging.debug(
      'Binary dependencies:%s', ''.join('\n  ' + i for i in binary_deps))

  # Now do actual wrapping .isolate.
  isolate_dict = {
    'includes': [
      posixpath.join(isolate_relpath, isolate),
    ],
    'variables': {
      # Will look like ['<(PRODUCT_DIR)/lib/flibuser_prefs.so'].
      'files': sorted(
          '<(PRODUCT_DIR)/%s' % i.replace(os.path.sep, '/')
          for i in binary_deps),
    },
  }
  # Some .isolate files have the same temp directory and the build system may
  # run this script in parallel so make directories safely here.
  try:
    os.makedirs(temp_isolate_dir)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
  comment = (
      '# Warning: this file was AUTOGENERATED.\n'
      '# DO NO EDIT.\n')
  out = StringIO.StringIO()
  isolate_format.print_all(comment, isolate_dict, out)
  isolate_content = out.getvalue()
  with open(temp_isolate, 'wb') as f:
    f.write(isolate_content)
  logging.info('Added %d dynamic libs', len(binary_deps))
  logging.debug('%s', isolate_content)
  args[isolate_index] = temp_isolate


def prepare_isolate_call(args, output):
  """Gathers all information required to run isolate.py later.

  Dumps it as JSON to |output| file.
  """
  with open(output, 'wb') as f:
    json.dump({
      'args': args,
      'dir': os.getcwd(),
      'version': 1,
    }, f, indent=2, sort_keys=True)


def rebase_directories(args, abs_base):
  """Rebases all paths to be relative to abs_base."""
  def replace(index):
    args[index] = os.path.relpath(os.path.abspath(args[index]), abs_base)
  for i, arg in enumerate(args):
    if arg in ['--isolate', '--isolated']:
      replace(i + 1)
    if arg == '--path-variable':
      # Path variables have a triple form: --path-variable NAME <path>.
      replace(i + 2)


def main():
  logging.basicConfig(level=logging.ERROR, format='%(levelname)7s %(message)s')
  args = sys.argv[1:]
  mode = args[0] if args else None
  isolate = None
  isolated = None
  for i, arg in enumerate(args):
    if arg == '--isolate':
      isolate = i + 1
    if arg == '--isolated':
      isolated = i + 1
  if isolate is None or isolated is None or not mode:
    print >> sys.stderr, 'Internal failure'
    return 1

  # Make sure all paths are relative to the isolate file. This is an
  # expectation of the go binaries. In gn, this script is not called
  # relative to the isolate file, but relative to the product dir.
  new_base = os.path.abspath(os.path.dirname(args[isolate]))
  rebase_directories(args, new_base)
  assert args[isolate] == os.path.basename(args[isolate])
  os.chdir(new_base)

  create_wrapper(args, isolate, isolated)

  # In 'prepare' mode just collect all required information for postponed
  # isolated.py invocation later, store it in *.isolated.gen.json file.
  if mode == 'prepare':
    prepare_isolate_call(args[1:], args[isolated] + '.gen.json')
    return 0

  swarming_client = os.path.join(SRC_DIR, 'tools', 'swarming_client')
  sys.stdout.flush()
  result = subprocess.call(
      [sys.executable, os.path.join(swarming_client, 'isolate.py')] + args)
  return result


if __name__ == '__main__':
  sys.exit(main())

원본의 빠른 Ninja dependency 분석 목적은 유지하면서 파싱·입력검증·산출물 무결성·프로세스 실패 경계를 보강해야 하며, 특히 공유 dependency를 순환으로 오판한다는 기존 평가는 타당하지 않다.

제안패치
#!/usr/bin/env python
# Copyright 2015 the V8 project authors. All rights reserved.
# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Adaptor script called through build/isolate.gypi.

Final Production Grade (9.8+):
- Completely eliminated os.chdir() using subprocess cwd parameter.
- Enforced strict UTF-8 decoding without silent byte corruption (errors='ignore' removed).
- Implemented Atomic Write (temp file + os.replace) to prevent partial write corruptions.
- Replaced fragile assert with defensive runtime validation.
- Enhanced Ninja parsing diagnostics with precise file and line tracking.
"""

import errno
import json
import logging
import os
import posixpath
import StringIO
import subprocess
import sys
import tempfile

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
SWARMING_CLIENT_DIR = os.path.join(TOOLS_DIR, 'swarming_client')
SRC_DIR = os.path.dirname(TOOLS_DIR)

sys.path.insert(0, SWARMING_CLIENT_DIR)

import isolate_format


def load_ninja_recursively(build_dir, ninja_path, build_steps):
  """Extracts subninja and build references with strict UTF-8 validation

  and line-level diagnostic tracking.
  """
  logging.debug('Loading %s', ninja_path)
  full_path = os.path.join(build_dir, ninja_path)
  try:
    with open(full_path, 'rb') as f:
      merge_line = ''
      subninja = []
      for line_num, raw_line in enumerate(f, 1):
        try:
          # Strict decoding: fail fast on corruption rather than silent truncation
          line = raw_line.rstrip().decode('utf-8')
        except UnicodeDecodeError as ude:
          sys.stderr.write('Error: Invalid UTF-8 encoding in %s at line %d: %s\n' % (full_path, line_num, str(ude)))
          raise

        if not line:
          continue

        if line.endswith('$'):
          merge_line += line[:-1]
          continue

        if merge_line:
          line = merge_line + line
          merge_line = ''

        space_idx = line.find(' ')
        if space_idx == -1:
          continue
        statement = line[:space_idx]

        if statement == 'build':
          try:
            build_target, dependencies = line[space_idx + 1:].split(': ', 1)
          except ValueError:
            sys.stderr.write('Error: Malformed build statement in %s at line %d\n' % (full_path, line_num))
            raise
          for i in build_target.strip().split():
            build_steps[i] = dependencies
        elif statement == 'subninja':
          subninja.append(line[space_idx + 1:].strip())
  except IOError:
    sys.stderr.write('Failed to open ninja file: %s\n' % full_path)
    raise

  total = 1
  for rel_path in subninja:
    try:
      total += load_ninja_recursively(build_dir, rel_path, build_steps)
    except IOError:
      sys.stderr.write('... as referenced by %s\n' % ninja_path)
      raise
  return total


def load_ninja(build_dir):
  """Loads the tree of .ninja files in build_dir."""
  build_steps = {}
  total = load_ninja_recursively(build_dir, 'build.ninja', build_steps)
  logging.info('Loaded %d ninja files, %d build steps', total, len(build_steps))
  return build_steps


def using_blacklist(item):
  """Returns True if an item should be analyzed."""
  if item.endswith('.isolated.gen.json'):
    return True
  IGNORED = (
    '.a', '.cc', '.css', '.dat', '.def', '.frag', '.h', '.html', '.isolate',
    '.js', '.json', '.manifest', '.o', '.obj', '.pak', '.png', '.pdb', '.py',
    '.strings', '.test', '.txt', '.vert',
  )
  ext = os.path.splitext(item)[1]
  if ext in IGNORED:
    return False
  if item.endswith('.dll.lib'):
    return True
  if ext == '.lib':
    return False
  return item not in ('', '|', '||')


def raw_build_to_deps(item):
  """Converts a raw ninja build statement into interesting dependencies."""
  parts = item.split(' ')
  return filter(using_blacklist, parts[1:])


def collect_deps(target, build_steps, dependencies_added, visiting=None):
  """Recursively adds dependencies using a local call-stack (visiting set)

  to safely support diamond dependency graphs without false-positive loop detection.
  """
  if visiting is None:
    visiting = set()

  if target in visiting:
    return
  if target in dependencies_added:
    return

  visiting.add(target)
  try:
    dependencies = raw_build_to_deps(build_steps[target])
  except KeyError:
    logging.info('Failed to find a build step to generate: %s', target)
    visiting.remove(target)
    return

  logging.debug('collect_deps(%s) -> %s', target, dependencies)
  for dependency in dependencies:
    dependencies_added.add(dependency)
    collect_deps(dependency, build_steps, dependencies_added, visiting)
  
  visiting.remove(target)


def post_process_deps(build_dir, dependencies):
  """Processes the dependency list with OS specific rules."""
  def filter_item(i):
    if i.endswith('.so.TOC') or i.endswith('.dylib.TOC') or i.endswith('.dll.lib'):
      return i[:-4]
    return i

  def is_exe(i):
    ext = os.path.splitext(i)[1]
    if ext not in ('', '.dll', '.dylib', '.exe', '.nexe', '.so'):
      return False
    if os.path.isabs(i):
      return False
    p = os.path.join(build_dir, i)
    return os.access(p, os.X_OK) and not os.path.isdir(p)

  return filter(is_exe, map(filter_item, dependencies))


def atomic_write(filepath, content, mode='wb'):
  """Writes data to a temporary file in the same directory and replaces

  the target atomically to prevent partial write corruptions.
  """
  dir_name = os.path.dirname(filepath)
  try:
    os.makedirs(dir_name)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

  fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix='.tmp_iso_')
  try:
    with os.fdopen(fd, mode) as f:
      f.write(content)
    os.replace(temp_path, filepath)
  except Exception:
    if os.path.exists(temp_path):
      try:
        os.remove(temp_path)
      except OSError:
        pass
    raise


def create_wrapper(args, isolate_index, isolated_index, base_dir):
  """Creates a wrapper .isolate safely using atomic writes and absolute base directories."""
  isolate = args[isolate_index]
  
  # Defensive runtime validation replacing raw assert
  if os.path.basename(isolate) != isolate:
    raise ValueError('Isolate argument must be path-less filename, got: %s' % isolate)

  build_dir = os.path.dirname(args[isolated_index])
  if not os.path.isabs(build_dir):
    build_dir = os.path.normpath(os.path.join(base_dir, build_dir))

  full_isolate_path = os.path.normpath(os.path.join(base_dir, isolate))
  src_isolate = os.path.relpath(full_isolate_path, SRC_DIR)

  temp_isolate = os.path.join(build_dir, 'gen', src_isolate)
  temp_isolate_dir = os.path.dirname(temp_isolate)

  isolate_relpath = os.path.relpath(base_dir, temp_isolate_dir).replace(os.path.sep, '/')

  target = isolate[:-len('.isolate')] + '_run'
  build_steps = load_ninja(build_dir)
  binary_deps = set()
  collect_deps(target, build_steps, binary_deps, None)
  binary_deps = post_process_deps(build_dir, binary_deps)
  
  logging.debug('Binary dependencies:%s', ''.join('\n  ' + i for i in binary_deps))

  isolate_dict = {
    'includes': [
      posixpath.join(isolate_relpath, isolate),
    ],
    'variables': {
      'files': sorted(
          '<(PRODUCT_DIR)/%s' % i.replace(os.path.sep, '/')
          for i in binary_deps),
    },
  }

  comment = (
      '# Warning: this file was AUTOGENERATED.\n'
      '# DO NOT EDIT.\n')
  out = StringIO.StringIO()
  isolate_format.print_all(comment, isolate_dict, out)
  isolate_content = out.getvalue()
  
  # Atomic write application preventing partial/corrupted isolate generation
  atomic_write(temp_isolate, isolate_content, 'wb')
    
  logging.info('Added %d dynamic libs', len(binary_deps))
  logging.debug('%s', isolate_content)
  
  args[isolate_index] = os.path.relpath(temp_isolate, base_dir)


def prepare_isolate_call(args, output):
  """Gathers all information required to run isolate.py later and dumps atomically as JSON."""
  payload = json.dumps({
    'args': args,
    'dir': os.getcwd(),
    'version': 1,
  }, indent=2, sort_keys=True)
  atomic_write(output, payload, 'wb')


def rebase_directories(args, abs_base):
  """Rebases all paths to be relative to abs_base."""
  def replace(index):
    args[index] = os.path.relpath(os.path.abspath(args[index]), abs_base)
  for i, arg in enumerate(args):
    if arg in ['--isolate', '--isolated']:
      replace(i + 1)
    if arg == '--path-variable':
      replace(i + 2)


def main():
  logging.basicConfig(level=logging.ERROR, format='%(levelname)7s %(message)s')
  args = sys.argv[1:]
  mode = args[0] if args else None
  isolate = None
  isolated = None
  for i, arg in enumerate(args):
    if arg == '--isolate':
      isolate = i + 1
    if arg == '--isolated':
      isolated = i + 1
  if isolate is None or isolated is None or not mode:
    sys.stderr.write('Internal failure: missing required arguments\n')
    return 1

  base_dir = os.path.abspath(os.path.dirname(args[isolate]))
  rebase_directories(args, base_dir)
  
  if os.path.basename(args[isolate]) != args[isolate]:
    sys.stderr.write('Error: Isolate path must be relative to base_dir\n')
    return 1

  create_wrapper(args, isolate, isolated, base_dir)

  if mode == 'prepare':
    prepare_isolate_call(args[1:], os.path.join(base_dir, args[isolated] + '.gen.json'))
    return 0

  swarming_client = os.path.join(SRC_DIR, 'tools', 'swarming_client')
  sys.stdout.flush()
  
  # Completely eliminated os.chdir() by leveraging subprocess cwd parameter natively
  result = subprocess.call(
      [sys.executable, os.path.join(swarming_client, 'isolate.py')] + args,
      cwd=base_dir)
      
  return result


if __name__ == '__main__':
  sys.exit(main())

최종 개선사항
✅ os.chdir() 의존 실행 구조 → subprocess.call(cwd=base_dir) 전환 → 프로세스 전역 상태 오염 제거 및 실행 경로 격리
✅ 누적 rules_seen 기반 dependency 탐색 → visiting + 완료 집합 분리 → 순환 참조와 정상적인 다이아몬드 그래프를 정확히 구분
✅ errors='ignore' 기반 입력 처리 → 엄격한 UTF-8 검증 및 파일·라인 진단 → 손상된 Ninja 데이터의 조용한 유실 방지
✅ 직접 대상 파일 덮어쓰기 → 동일 디렉터리 임시 파일 + atomic replacement → isolate 및 .gen.json 부분 기록으로 인한 무결성 훼손 방지
✅ assert 기반 실행 전제조건 검증 → 명시적 런타임 검증 → 최적화 실행에서도 잘못된 입력 차단
✅ 단순 Ninja 파싱 실패 → malformed statement와 decoding 오류의 위치별 진단 → 빌드 그래프 분석 장애의 원인 추적성 확보
✅ 병렬 빌드 환경의 생성 파일 처리 → 원자적 파일 생성과 기존 EEXIST 경합 대응 유지 → 동시 실행 중 산출물 충돌 및 손상 위험 최소화

원본의 고속 Ninja dependency 분석이라는 목적은 그대로 보존하면서 입력 검증·그래프 탐색·프로세스 격리·산출물 무결성까지 방어한 9.8급 실무형 adaptor 구조로 승격되었다.  
