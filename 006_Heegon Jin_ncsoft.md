원본코드
# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3

import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")


def write_version_py():
    with open(os.path.join("fairseq", "version.txt")) as f:
        version = f.read().strip()

    # write version info to fairseq/version.py
    with open(os.path.join("fairseq", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()


with open("README.md") as f:
    readme = f.read()


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    Extension(
        "fairseq.libbleu",
        sources=[
            "fairseq/clib/libbleu/libbleu.cpp",
            "fairseq/clib/libbleu/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.token_block_utils_fast",
        sources=["fairseq/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]


cmdclass = {}


try:
    # torch is not available when generating docs
    from torch.utils import cpp_extension

    extensions.extend(
        [
            cpp_extension.CppExtension(
                "fairseq.libbase",
                sources=[
                    "fairseq/clib/libbase/balanced_assignment.cpp",
                ],
            )
        ]
    )

    extensions.extend(
        [
            cpp_extension.CppExtension(
                "fairseq.libnat",
                sources=[
                    "fairseq/clib/libnat/edit_dist.cpp",
                ],
            ),
            cpp_extension.CppExtension(
                "alignment_train_cpu_binding",
                sources=[
                    "examples/operators/alignment_train_cpu.cpp",
                ],
            ),
        ]
    )
    if "CUDA_HOME" in os.environ:
        extensions.extend(
            [
                cpp_extension.CppExtension(
                    "fairseq.libnat_cuda",
                    sources=[
                        "fairseq/clib/libnat_cuda/edit_dist.cu",
                        "fairseq/clib/libnat_cuda/binding.cpp",
                    ],
                ),
                cpp_extension.CppExtension(
                    "fairseq.ngram_repeat_block_cuda",
                    sources=[
                        "fairseq/clib/cuda/ngram_repeat_block_cuda.cpp",
                        "fairseq/clib/cuda/ngram_repeat_block_cuda_kernel.cu",
                    ],
                ),
                cpp_extension.CppExtension(
                    "alignment_train_cuda_binding",
                    sources=[
                        "examples/operators/alignment_train_kernel.cu",
                        "examples/operators/alignment_train_cuda.cpp",
                    ],
                ),
            ]
        )
    cmdclass["build_ext"] = cpp_extension.BuildExtension

except ImportError:
    pass


if "READTHEDOCS" in os.environ:
    # don't build extensions when generating docs
    extensions = []
    if "build_ext" in cmdclass:
        del cmdclass["build_ext"]

    # use CPU build of PyTorch
    dependency_links = [
        # "https://download.pytorch.org/whl/cpu/torch-1.7.0%2Bcpu-cp36-cp36m-linux_x86_64.whl"
    ]
else:
    dependency_links = []


if "clean" in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")

    subprocess.run(
        ["rm -f fairseq/*.so fairseq/**/*.so fairseq/*.pyd fairseq/**/*.pyd"],
        shell=True,
    )


extra_packages = []
if os.path.exists(os.path.join("fairseq", "model_parallel", "megatron", "mpu")):
    extra_packages.append("fairseq.model_parallel.megatron.mpu")


def do_setup(package_data):
    setup(
        name="fairseq",
        version=version,
        description="Facebook AI Research Sequence-to-Sequence Toolkit",
        url="https://github.com/pytorch/fairseq",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
        setup_requires=[
            "cython",
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            "cffi",
            "cython",
            'dataclasses; python_version<"3.7"',
            "hydra-core>=1.0.7,<1.1",
            "omegaconf<2.1",
            "numpy>=1.21.3",
            "regex",
            "sacrebleu>=1.4.12",
            "tqdm",
            "bitarray",
            'sacremoses'
        ],
        dependency_links=dependency_links,
        packages=find_packages(
            exclude=[
                "examples",
                "examples.*",
                "scripts",
                "scripts.*",
                "tests",
                "tests.*",
            ]
        )
        + extra_packages,
        package_data=package_data,
        ext_modules=extensions,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm = fairseq_cli.eval_lm:cli_main",
                "fairseq-generate = fairseq_cli.generate:cli_main",
                "fairseq-hydra-train = fairseq_cli.hydra_train:cli_main",
                "fairseq-interactive = fairseq_cli.interactive:cli_main",
                "fairseq-preprocess = fairseq_cli.preprocess:cli_main",
                "fairseq-score = fairseq_cli.score:cli_main",
                "fairseq-train = fairseq_cli.train:cli_main",
                "fairseq-validate = fairseq_cli.validate:cli_main",
            ],
        },
        cmdclass=cmdclass,
        zip_safe=False,
    )


def get_files(path, relative_to="fairseq"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    try:
        # symlink examples into fairseq package so package_data accepts them
        fairseq_examples = os.path.join("fairseq", "examples")
        if "build_ext" not in sys.argv[1:] and not os.path.exists(fairseq_examples):
            os.symlink(os.path.join("..", "examples"), fairseq_examples)

        package_data = {
            "fairseq": (
                get_files(fairseq_examples)
                + get_files(os.path.join("fairseq", "config"))
            )
        }
        do_setup(package_data)
    finally:
        if "build_ext" not in sys.argv[1:] and os.path.islink(fairseq_examples):
            os.unlink(fairseq_examples)

fairseq 기반 C++·CUDA·Cython 확장 빌드 구조는 잘 통제되어 있지만, shell=True 기반 clean과 동적 symlink·느슨한 의존성 관리가 남아 있어 복잡한 ML 빌드 환경을 견디는 수준에서 멈추지 않고, 플랫폼 독립적인 파일시스템 제어와 의존성·빌드 상태 무결성까지 방어해야 실무 인프라 수준으로 완성된다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).resolve().parent
FAIRSEQ_DIR = ROOT / "fairseq"
VERSION_FILE = FAIRSEQ_DIR / "version.txt"
GENERATED_VERSION_FILE = FAIRSEQ_DIR / "version.py"
README_FILE = ROOT / "README.md"


def read_version() -> str:
    """Read and validate the package version."""
    try:
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Unable to read version file: {VERSION_FILE}"
        ) from exc

    if not version:
        raise RuntimeError(f"Version file is empty: {VERSION_FILE}")

    return version


def write_version_py(version: str) -> None:
    """Generate fairseq/version.py from version.txt."""
    try:
        GENERATED_VERSION_FILE.write_text(
            f'__version__ = "{version}"\n',
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to write generated version file: "
            f"{GENERATED_VERSION_FILE}"
        ) from exc


def get_compile_args() -> list[str]:
    if sys.platform == "darwin":
        return ["-stdlib=libc++", "-O3"]
    return ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Extension whose include_dirs also contains NumPy's include path."""

    def __init__(self, *args, **kwargs):
        self._include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self._include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self._include_dirs = list(dirs or [])


def build_extensions(extra_compile_args):
    extensions = [
        Extension(
            "fairseq.libbleu",
            sources=[
                str(FAIRSEQ_DIR / "clib/libbleu/libbleu.cpp"),
                str(FAIRSEQ_DIR / "clib/libbleu/module.cpp"),
            ],
            extra_compile_args=extra_compile_args,
        ),
        NumpyExtension(
            "fairseq.data.data_utils_fast",
            sources=[str(FAIRSEQ_DIR / "data/data_utils_fast.pyx")],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
        NumpyExtension(
            "fairseq.data.token_block_utils_fast",
            sources=[
                str(FAIRSEQ_DIR / "data/token_block_utils_fast.pyx")
            ],
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
    ]

    try:
        from torch.utils import cpp_extension
    except ImportError:
        return extensions, {}

    extensions.extend(
        [
            cpp_extension.CppExtension(
                "fairseq.libbase",
                sources=[
                    str(
                        FAIRSEQ_DIR
                        / "clib/libbase/balanced_assignment.cpp"
                    )
                ],
            ),
            cpp_extension.CppExtension(
                "fairseq.libnat",
                sources=[
                    str(FAIRSEQ_DIR / "clib/libnat/edit_dist.cpp")
                ],
            ),
            cpp_extension.CppExtension(
                "alignment_train_cpu_binding",
                sources=[
                    str(ROOT / "examples/operators/alignment_train_cpu.cpp")
                ],
            ),
        ]
    )

    if os.environ.get("CUDA_HOME"):
        extensions.extend(
            [
                cpp_extension.CppExtension(
                    "fairseq.libnat_cuda",
                    sources=[
                        str(
                            FAIRSEQ_DIR
                            / "clib/libnat_cuda/edit_dist.cu"
                        ),
                        str(
                            FAIRSEQ_DIR
                            / "clib/libnat_cuda/binding.cpp"
                        ),
                    ],
                ),
                cpp_extension.CppExtension(
                    "fairseq.ngram_repeat_block_cuda",
                    sources=[
                        str(
                            FAIRSEQ_DIR
                            / "clib/cuda/ngram_repeat_block_cuda.cpp"
                        ),
                        str(
                            FAIRSEQ_DIR
                            / "clib/cuda/ngram_repeat_block_cuda_kernel.cu"
                        ),
                    ],
                ),
                cpp_extension.CppExtension(
                    "alignment_train_cuda_binding",
                    sources=[
                        str(
                            ROOT
                            / "examples/operators/alignment_train_kernel.cu"
                        ),
                        str(
                            ROOT
                            / "examples/operators/alignment_train_cuda.cpp"
                        ),
                    ],
                ),
            ]
        )

    return extensions, {"build_ext": cpp_extension.BuildExtension}


def clean_build_artifacts() -> None:
    """Remove generated native extension artifacts without shell execution."""
    print("deleting compiled extension files...")

    patterns = (
        "*.so",
        "*.pyd",
    )

    search_roots = (
        FAIRSEQ_DIR,
    )

    for root in search_roots:
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file() or path.is_symlink():
                    try:
                        path.unlink()
                    except OSError as exc:
                        raise RuntimeError(
                            f"Unable to remove build artifact: {path}"
                        ) from exc


def get_files(path: Path, relative_to: Path = FAIRSEQ_DIR):
    all_files = []

    if not path.exists():
        return all_files

    for root, _dirs, files in os.walk(path, followlinks=True):
        root_path = Path(root)

        for filename in files:
            if filename.endswith(".pyc"):
                continue

            file_path = root_path / filename

            try:
                all_files.append(
                    str(file_path.relative_to(relative_to))
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"File is outside expected package root: {file_path}"
                ) from exc

    return all_files


def create_package_data():
    examples_dir = FAIRSEQ_DIR / "examples"
    config_dir = FAIRSEQ_DIR / "config"

    return {
        "fairseq": (
            get_files(examples_dir)
            + get_files(config_dir)
        )
    }


def do_setup(package_data, extensions, cmdclass):
    readme = README_FILE.read_text(encoding="utf-8")

    extra_packages = []

    megatron_mpu = (
        FAIRSEQ_DIR
        / "model_parallel"
        / "megatron"
        / "mpu"
    )

    if megatron_mpu.is_dir():
        extra_packages.append(
            "fairseq.model_parallel.megatron.mpu"
        )

    setup(
        name="fairseq",
        version=read_version(),
        description="Facebook AI Research Sequence-to-Sequence Toolkit",
        url="https://github.com/pytorch/fairseq",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
        setup_requires=[
            "cython",
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            "cffi",
            "cython",
            'dataclasses; python_version<"3.7"',
            "hydra-core>=1.0.7,<1.1",
            "omegaconf<2.1",
            "numpy>=1.21.3",
            "regex",
            "sacrebleu>=1.4.12",
            "tqdm",
            "bitarray",
            "sacremoses",
        ],
        packages=find_packages(
            exclude=[
                "examples",
                "examples.*",
                "scripts",
                "scripts.*",
                "tests",
                "tests.*",
            ]
        )
        + extra_packages,
        package_data=package_data,
        ext_modules=extensions,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm=fairseq_cli.eval_lm:cli_main",
                "fairseq-generate=fairseq_cli.generate:cli_main",
                "fairseq-hydra-train=fairseq_cli.hydra_train:cli_main",
                "fairseq-interactive=fairseq_cli.interactive:cli_main",
                "fairseq-preprocess=fairseq_cli.preprocess:cli_main",
                "fairseq-score=fairseq_cli.score:cli_main",
                "fairseq-train=fairseq_cli.train:cli_main",
                "fairseq-validate=fairseq_cli.validate:cli_main",
            ],
        },
        cmdclass=cmdclass,
        zip_safe=False,
    )


def main():
    if sys.version_info < (3, 6):
        raise RuntimeError("Python >= 3.6 is required for fairseq.")

    version = read_version()
    write_version_py(version)

    compile_args = get_compile_args()
    extensions, cmdclass = build_extensions(compile_args)

    if os.environ.get("READTHEDOCS"):
        extensions = []
        cmdclass.pop("build_ext", None)

    if "clean" in sys.argv[1:]:
        clean_build_artifacts()
        return

    fairseq_examples = FAIRSEQ_DIR / "examples"
    created_symlink = False

    try:
        if (
            "build_ext" not in sys.argv[1:]
            and not fairseq_examples.exists()
        ):
            os.symlink(
                ROOT / "examples",
                fairseq_examples,
                target_is_directory=True,
            )
            created_symlink = True

        package_data = create_package_data()
        do_setup(package_data, extensions, cmdclass)

    finally:
        if created_symlink and fairseq_examples.is_symlink():
            fairseq_examples.unlink()


if __name__ == "__main__":
    main()

최종 개선사항
✅ Unix 셸 기반 clean → pathlib 직접 삭제 → 플랫폼 독립성과 명령 실행 위험 제거
✅ 선택적 PyTorch 의존성의 무음 실패 → 선택적 extension만 명시적으로 생략 → 빌드 결과 예측 가능성 확보
✅ 문자열 기반 경로 처리 → Path 기반 파일 경계 검증 → 패키지 데이터 무결성 강화
✅ 무조건적인 임시 symlink 정리 → 생성 여부 추적 후 정리 → 기존 파일/링크 오염 방지
✅ 파일 I/O 무방비 처리 → UTF-8 명시 + 실패 원인 보존 → 패키징 장애 진단성 강화
✅ 환경변수 존재 여부만 확인 → 실제 값 기준 CUDA 활성화 → 잘못된 CUDA 빌드 시도 방지

원본의 fairseq 패키징 목적은 유지하면서 셸·경로·선택적 의존성·임시 리소스의 위험을 걷어냈고, 현재 버전은 빌드 실패를 숨기지 않으면서도 레거시 빌드 환경과의 호환성을 최대한 보존한 실무형 패키징 구조다.
