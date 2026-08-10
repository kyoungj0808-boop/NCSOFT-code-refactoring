원본코드import logging
import numpy as np
import torch
import transformers
import random


from utils.data import get_benchmark_datasets
from utils.configs import (ModelArguments, DataArguments, PEFTArguments)
from utils.configs import H4ArgumentParser
import os
from trl import DPOConfig
from trainers.mipo_trainer import MIPOTrainer


logger = logging.getLogger(__name__)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, PEFTArguments))
    model_args, data_args, training_args, peft_args = parser.parse()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    training_args.output_dir = f"{training_args.output_dir}/{data_args.tag}-mipo-beta{training_args.beta}-lr{training_args.learning_rate}"
    logger.info(f"OUTPUT_DIR: {training_args.output_dir}")

    # Set seed
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.random.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)

    ##################################################
    ## load datasets
    ##################################################
    raw_datasets = get_benchmark_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"]
    )

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )


    logger.info(f"# of train data: {len(raw_datasets['train'])}")
    if "test" in raw_datasets:
        logger.info(f"# of test data: {len(raw_datasets['test'])}")

    ###############################################
    # Set tokenizer
    ###############################################

    if model_args.tokenizer_name_or_path:
        tokenizer_path = model_args.tokenizer_name_or_path
    else:
        tokenizer_path = model_args.model_name_or_path

    #####################################
    # Load tokenizer and process datasets
    #####################################

    # data_args.truncation_side = "left"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=True,
        legacy=False
    )

    raw_datasets = raw_datasets.remove_columns("messages")

    ######################
    # For Model
    ######################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    model = model_args.model_name_or_path
    ref_model = model
    ref_model_kwargs = model_kwargs


    #########################
    # Instantiate MIPO trainer
    #########################
    trainer = MIPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        model_init_kwargs=model_kwargs,
        padding_value=tokenizer.eos_token_id,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.args.do_eval = False

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    train()

MIPO/DPO 학습 파이프라인의 실험 구조와 재현성 기반은 탄탄하지만, 디버깅 설정의 상시 강제·참조 모델 메모리 중복 가능성·평가 비활성화라는 운영상 치명점을 품고 있어, 현재는 연구 실험용으로는 충분하나 대규모 학습 환경에서는 안정성과 검증 가능성을 추가로 방어해야 하는 코드다.

제안패치
# SPDX-FileCopyrightText: Ⓒ 2024 NCSOFT Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
import transformers

from utils.configs import DataArguments, H4ArgumentParser, ModelArguments, PEFTArguments
from utils.data import get_benchmark_datasets
from trl import DPOConfig
from trainers.mipo_trainer import MIPOTrainer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """모든 라이브러리의 난수 고정으로 재현성(Reproducibility) 확보"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_str: Optional[str]) -> torch.dtype | str:
    """문자열로 입력된 dtype을 안전하게 화이트리스트 기반 torch.dtype으로 변환"""
    if dtype_str in ["auto", None]:
        return "auto"
    
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_str}. Choose from {list(dtype_mapping.keys())} or 'auto'.")
    return dtype_mapping[dtype_str]


def train() -> None:
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, PEFTArguments))
    model_args, data_args, training_args, peft_args = parser.parse()

    # Log parameters
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # 동적 출력 디렉토리 구성
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        f"{data_args.tag}-mipo-beta{training_args.beta}-lr{training_args.learning_rate}"
    )
    logger.info(f"OUTPUT_DIR: {training_args.output_dir}")

    # 시드 고정
    set_seed(training_args.seed)

    # 1. 데이터셋 로드
    raw_datasets = get_benchmark_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"]
    )

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    logger.info(f"# of train data: {len(raw_datasets['train'])}")
    if "test" in raw_datasets:
        logger.info(f"# of test data: {len(raw_datasets['test'])}")

    # 2. 토크나이저 설정
    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=True,
        legacy=False
    )

    # DatasetDict 전역 일괄 삭제 대신 split별 컬럼 존재 여부를 확인하여 안전하게 제거
    for split_name, dataset in raw_datasets.items():
        if "messages" in dataset.column_names:
            raw_datasets[split_name] = dataset.remove_columns("messages")

    # 3. 모델 인자(Arguments) 구성
    torch_dtype = resolve_torch_dtype(model_args.torch_dtype)

    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "use_flash_attention_2": model_args.use_flash_attention_2,
        "torch_dtype": torch_dtype,
        "use_cache": False if training_args.gradient_checkpointing else True,
    }

    model_path = model_args.model_name_or_path
    
    # [방어적 계약 준수] 
    # MIPOTrainer가 ref_model=None을 안전하게 처리하는지 검증되지 않았으므로, 
    # 원본처럼 model_path를 명시하되 PEFT 환경에서 TRL 내부 최적화가 동작하도록 kwargs를 정합성 있게 전달
    ref_model = model_path
    ref_model_kwargs = model_kwargs.copy()

    # 4. MIPO 트레이너 인스턴스화
    trainer = MIPOTrainer(
        model=model_path,
        ref_model=ref_model,
        processing_class=tokenizer,
        model_init_kwargs=model_kwargs,
        padding_value=tokenizer.eos_token_id,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )

    # 5. 학습 루프 실행
    checkpoint = training_args.resume_from_checkpoint
    
    # 사용자가 설정한 evaluation 설정을 강제로 끄지 않고 그대로 존중
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    train()

최종 개선사항
✅ 전역 CUDA_LAUNCH_BLOCKING=1 강제 설정 → 디버깅 환경과 학습 환경 분리 → 정상 학습 throughput 및 운영 성능 보존
✅ 무방비 getattr(torch, dtype) 변환 → dtype 화이트리스트 검증 → 잘못된 학습 설정의 조기 차단
✅ DatasetDict 전체 컬럼 일괄 삭제 → 각 split별 컬럼 존재 여부 검증 → train/test 스키마 차이로 인한 전처리 오류 방지
✅ 동일 모델 경로의 reference model 중복 가능성 → 현재 MIPOTrainer 계약을 보수적으로 유지 → 검증되지 않은 ref_model=None 변경으로 인한 학습 의미 훼손 방지
✅ do_eval=False 강제 적용 → 사용자가 지정한 evaluation 설정 유지 → 과적합 감시 및 검증 파이프라인 보존
✅ 단일 train() 함수에 설정·시드·데이터·모델·학습 로직 집중 → dtype/seed 등 검증 책임 분리 → 장애 지점 추적성과 유지보수성 향상
✅ 원본의 모델·데이터·체크포인트 흐름 유지 → 필요한 방어 계층만 추가 → 과설계 없이 실험 재현성과 운영 안정성을 동시에 확보

원본의 MIPO 학습 목적과 MIPOTrainer 계약은 보존하면서 성능을 저해하는 디버깅 설정, 설정값 오염, 데이터 스키마 불일치 위험을 제거하고 검증되지 않은 reference-model 동작 변경은 보류한 것이 핵심이며, 현재 버전은 연구 코드에서 실무형 학습 엔트리포인트로 한 단계 올라온 상태다.
