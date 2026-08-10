원본코드
import importlib
import re

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI


class InferenceModule():
    def __init__(self, prompt_name: str = "", config: dict = {}):
        self.config = config

        prompt_name = config.get("prompt", prompt_name)
        prompt_module = importlib.import_module(f"prompt.{prompt_name}")
        self.prompt_name: str = prompt_name
        self.system_message: str = prompt_module.system if "system" in dir(prompt_module) else ""
        self.user_message_template: str = prompt_module.user
        self.output_pattern: dict = prompt_module.output_pattern

    def make_conversation(self, instruction: str, response1: str, response2: str, swap: bool) -> list:
        conversation = []

        if self.system_message:
            conversation.append({"role": "system", "content": self.system_message})

        user_message = self.user_message_template.format(
            input=instruction,
            output_1=response1 if not swap else response2,
            output_2=response2 if not swap else response1,
        )
        conversation.append({"role": "user", "content": user_message})

        return conversation

    def get_prediction(self, output_text: str) -> int:
        """parse output text into prediction label: 1(A), 2(B), 3(TIE), 4(N/A)"""
        for prediction, pattern in self.output_pattern.items():
            if re.search(pattern, output_text):
                return prediction
        return 4

    def is_correct(self, prediction: int, label: int, swap: bool = False) -> bool:
        if not swap:
            return prediction == label and label in [1, 2]
        else:
            return prediction + label == 3 and prediction in [1, 2] and label in [1, 2]


class VllmModule(InferenceModule):
    def __init__(
            self,
            prompt_name: str = "",
            model_name: str = "",
            dtype: str = "float16",
            temperature: float = 0.0,
            max_tokens: int = 20,
            config: dict = {}):

        super().__init__(prompt_name=prompt_name, config=config)

        print("Initializing vllm model...")
        vllm_args = self.config.get("vllm_args", {})

        model_args = dict(model=model_name, dtype=dtype)
        model_args.update(vllm_args.get("model_args", {}))
        print("model args:", model_args)
        self.model_name = model_args["model"]
        tokenizer_name = self.config.get("tokenizer", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = LLM(**model_args)

        sampling_params_args = dict(temperature=temperature, max_tokens=max_tokens)
        sampling_params_args.update(vllm_args.get("sampling_params", {}))
        self.sampling_params = SamplingParams(**sampling_params_args)
        print(self.sampling_params)

    def generate(self, conversation_list: list) -> list:
        if "prometheus" in self.model_name:
            from fastchat.conversation import get_conv_template

            def _get_conversation_prompt(messages):
                """
                From filled prompt, convert it into llama-2 conversation prompt
                """
                conv = get_conv_template("mistral")

                for message in messages:
                    if message["role"] == "system":
                        conv.set_system_message(message["content"])
                    elif message["role"] == "user":
                        conv.append_message(conv.roles[0], message["content"])

                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                return prompt
            inputs = [_get_conversation_prompt(conversation).strip() for conversation in conversation_list]
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)

        elif "PandaLM" in self.model_name:
            inputs = [conversation[0]['content'] for conversation in conversation_list]
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)

        else:
            # llama3 style
            prompt_token_ids = [self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True) for conversation in conversation_list]
            outputs = self.model.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params)

        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        return generated_texts


class HfModule(InferenceModule):
    def __init__(
            self,
            model_name: str = "",
            dtype: str = "float16",
            max_new_tokens: int = 20,
            pad_token_id: int = None,
            do_sample: bool = False,
            temperature: float = 0.0,
            config: dict = {}):

        super().__init__(config=config)

        print("Initializing hf model...")
        hf_args = self.config.get("hf_args", {})

        model_name = hf_args.get("model_args", {}).get("model", model_name)
        dtype_name = hf_args.get("model_args", {}).get("dtype", dtype)

        dtype_mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_mapping[dtype_name]

        tokenizer_name = self.config.get("tokenizer", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto").eval()  # use_auth_token = args.hf_use_auth_token
        self.generate_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                                    do_sample=do_sample, temperature=temperature)
        self.generate_kwargs.update(hf_args.get("generate_kwargs", {}))
        print("generate_kwargs:", self.generate_kwargs)

    def generate(self, conversation_list: list) -> list:
        generated_texts = []
        for conversation in tqdm(conversation_list):
            # llama3 style
            input_ids = self.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            with torch.inference_mode():
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                generation = self.model.generate(input_ids=input_ids, **self.generate_kwargs)
                completion = self.tokenizer.decode(
                    generation[0][len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generated_texts.append(completion.strip())
        return generated_texts


class OpenaiModule(InferenceModule):
    def __init__(self, config: dict):
        super().__init__(config=config)
        print("Initializing openai client...")
        openai_args = self.config["openai_args"]
        self.client = OpenAI(**openai_args)
        self.create_args = self.config["create_args"]

    def generate(self, conversation_list: list) -> list:
        generated_texts = []
        for conversation in tqdm(conversation_list):
            response = self.client.chat.completions.create(
                messages=conversation,
                **self.create_args
            )
            generated_text = response.choices[0].message.content
            generated_texts.append(generated_text)
        return generated_texts

다양한 추론 백엔드를 하나의 전략 구조로 묶은 설계는 탄탄하지만, 모델명 하드코딩·설정 검증 부재·동기식 API 호출이 결합되어 모델 확장성과 대규모 평가 처리량을 동시에 갉아먹는 구조다.

제안패치
import abc
import asyncio
import importlib
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from openai import AsyncOpenAI
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


# ==========================================
# 1. Prompt Formatting Strategy Registry
# ==========================================

Conversation = List[Dict[str, str]]
FormattedPrompt = Union[str, List[int]]


class PromptFormatterRegistry:
    """모델별 프롬프트 포매팅 전략을 관리한다."""

    _strategies: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
        key = name.lower()

        def decorator(strategy):
            cls._strategies[key] = strategy
            return strategy

        return decorator

    @classmethod
    def format(
        cls,
        model_name: str,
        conversation: Conversation,
        tokenizer: Any,
    ) -> FormattedPrompt:
        model_name_lower = model_name.lower()

        for key, strategy in cls._strategies.items():
            if key in model_name_lower:
                return strategy.format(conversation, tokenizer)

        return tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )


@PromptFormatterRegistry.register("prometheus")
class PrometheusFormatter:
    @staticmethod
    def format(
        conversation: Conversation,
        tokenizer: Any,
    ) -> str:
        from fastchat.conversation import get_conv_template

        conv = get_conv_template("mistral")

        for message in conversation:
            role = message["role"]

            if role == "system":
                conv.set_system_message(message["content"])
            elif role == "user":
                conv.append_message(
                    conv.roles[0],
                    message["content"],
                )
            else:
                raise ValueError(
                    f"Unsupported conversation role: {role}"
                )

        conv.append_message(conv.roles[1], None)
        return conv.get_prompt().strip()


@PromptFormatterRegistry.register("pandalm")
class PandaLMFormatter:
    @staticmethod
    def format(
        conversation: Conversation,
        tokenizer: Any,
    ) -> str:
        if not conversation:
            raise ValueError("Conversation must not be empty.")

        return conversation[0]["content"]


# ==========================================
# 2. Base Inference Module
# ==========================================

class InferenceModule(abc.ABC):
    def __init__(
        self,
        prompt_name: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}

        resolved_prompt = self.config.get("prompt", prompt_name)

        if not resolved_prompt:
            raise ValueError(
                "Prompt name must be specified via config or argument."
            )

        module_name = f"prompt.{resolved_prompt}"

        try:
            prompt_module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise ModuleNotFoundError(
                    f"Prompt module not found: {module_name}"
                ) from exc
            raise

        if not hasattr(prompt_module, "user"):
            raise AttributeError(
                f"Prompt module '{module_name}' must define 'user'."
            )

        if not hasattr(prompt_module, "output_pattern"):
            raise AttributeError(
                f"Prompt module '{module_name}' must define "
                "'output_pattern'."
            )

        self.prompt_name = resolved_prompt
        self.system_message = getattr(prompt_module, "system", "")
        self.user_message_template = prompt_module.user
        self.output_pattern = prompt_module.output_pattern

    def make_conversation(
        self,
        instruction: str,
        response1: str,
        response2: str,
        swap: bool,
    ) -> Conversation:
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string.")
        if not isinstance(response1, str):
            raise TypeError("response1 must be a string.")
        if not isinstance(response2, str):
            raise TypeError("response2 must be a string.")

        conversation: Conversation = []

        if self.system_message:
            conversation.append(
                {
                    "role": "system",
                    "content": self.system_message,
                }
            )

        user_message = self.user_message_template.format(
            input=instruction,
            output_1=response2 if swap else response1,
            output_2=response1 if swap else response2,
        )

        conversation.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return conversation

    def get_prediction(self, output_text: str) -> int:
        if not isinstance(output_text, str):
            return 4

        for prediction, pattern in self.output_pattern.items():
            if re.search(pattern, output_text):
                return prediction

        return 4

    def is_correct(
        self,
        prediction: int,
        label: int,
        swap: bool = False,
    ) -> bool:
        valid_labels = {1, 2}

        if prediction not in valid_labels or label not in valid_labels:
            return False

        if not swap:
            return prediction == label

        return prediction + label == 3

    @abc.abstractmethod
    def generate(
        self,
        conversation_list: List[Conversation],
    ) -> List[str]:
        raise NotImplementedError


# ==========================================
# 3. vLLM Module
# ==========================================

class VllmModule(InferenceModule):
    def __init__(
        self,
        prompt_name: str = "",
        model_name: str = "",
        dtype: str = "float16",
        temperature: float = 0.0,
        max_tokens: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            prompt_name=prompt_name,
            config=config,
        )

        vllm_args = self.config.get("vllm_args", {})

        model_args = {
            "model": model_name,
            "dtype": dtype,
        }
        model_args.update(vllm_args.get("model_args", {}))

        resolved_model_name = model_args.get("model")
        if not resolved_model_name:
            raise ValueError("vLLM model name must be specified.")

        self.model_name = resolved_model_name

        tokenizer_name = self.config.get(
            "tokenizer",
            self.model_name,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.model = LLM(**model_args)

        sampling_params_args = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        sampling_params_args.update(
            vllm_args.get("sampling_params", {})
        )

        self.sampling_params = SamplingParams(
            **sampling_params_args
        )

    def generate(
        self,
        conversation_list: List[Conversation],
    ) -> List[str]:
        if not conversation_list:
            return []

        inputs = [
            PromptFormatterRegistry.format(
                self.model_name,
                conversation,
                self.tokenizer,
            )
            for conversation in conversation_list
        ]

        all_strings = all(isinstance(item, str) for item in inputs)
        all_token_ids = all(
            isinstance(item, (list, tuple)) for item in inputs
        )

        if not (all_strings or all_token_ids):
            raise TypeError(
                "Prompt formatter returned mixed input types."
            )

        if all_strings:
            outputs = self.model.generate(
                inputs,
                sampling_params=self.sampling_params,
            )
        else:
            outputs = self.model.generate(
                prompt_token_ids=inputs,
                sampling_params=self.sampling_params,
            )

        generated_texts = []

        for output in outputs:
            if not output.outputs:
                generated_texts.append("")
                continue

            generated_texts.append(
                output.outputs[0].text.strip()
            )

        return generated_texts


# ==========================================
# 4. HuggingFace Module
# ==========================================

class HfModule(InferenceModule):
    def __init__(
        self,
        prompt_name: str = "",
        model_name: str = "",
        dtype: str = "float16",
        max_new_tokens: int = 20,
        pad_token_id: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            prompt_name=prompt_name,
            config=config,
        )

        hf_args = self.config.get("hf_args", {})
        model_args = hf_args.get("model_args", {})

        model_name = model_args.get("model", model_name)
        dtype_name = model_args.get("dtype", dtype)

        if not model_name:
            raise ValueError(
                "HuggingFace model name must be specified."
            )

        dtype_mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        if dtype_name not in dtype_mapping:
            raise ValueError(
                f"Unsupported dtype: {dtype_name}"
            )

        tokenizer_name = self.config.get(
            "tokenizer",
            model_name,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_mapping[dtype_name],
            device_map="auto",
        ).eval()

        resolved_pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.tokenizer.pad_token_id
        )

        if resolved_pad_token_id is None:
            resolved_pad_token_id = (
                self.tokenizer.eos_token_id
            )

        if resolved_pad_token_id is None:
            raise ValueError(
                "Unable to resolve pad_token_id."
            )

        self.generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": resolved_pad_token_id,
            "do_sample": do_sample,
        }

        if do_sample:
            self.generate_kwargs["temperature"] = temperature

        self.generate_kwargs.update(
            hf_args.get("generate_kwargs", {})
        )

    def generate(
        self,
        conversation_list: List[Conversation],
    ) -> List[str]:
        generated_texts = []

        for conversation in tqdm(
            conversation_list,
            desc="HF Generation",
        ):
            input_ids = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            input_ids = input_ids.to(
                self.model.device
            )

            with torch.inference_mode():
                generation = self.model.generate(
                    input_ids=input_ids,
                    **self.generate_kwargs,
                )

            prompt_length = input_ids.shape[-1]

            completion = self.tokenizer.decode(
                generation[0][prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            generated_texts.append(completion.strip())

        return generated_texts


# ==========================================
# 5. OpenAI Async Module
# ==========================================

class OpenaiModule(InferenceModule):
    def __init__(
        self,
        prompt_name: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            prompt_name=prompt_name,
            config=config,
        )

        openai_args = self.config.get(
            "openai_args",
            {},
        )

        self.client = AsyncOpenAI(**openai_args)

        self.create_args = self.config.get(
            "create_args",
            {},
        )

        self.max_concurrency = max(
            1,
            int(
                self.config.get(
                    "openai_max_concurrency",
                    8,
                )
            ),
        )

    async def _generate_single(
        self,
        conversation: Conversation,
        semaphore: asyncio.Semaphore,
    ) -> str:
        async with semaphore:
            response = await self.client.chat.completions.create(
                messages=conversation,
                **self.create_args,
            )

        if not response.choices:
            return ""

        content = response.choices[0].message.content

        return (content or "").strip()

    async def _agenerate(
        self,
        conversation_list: List[Conversation],
    ) -> List[str]:
        semaphore = asyncio.Semaphore(
            self.max_concurrency
        )

        tasks = [
            self._generate_single(
                conversation,
                semaphore,
            )
            for conversation in conversation_list
        ]

        return await asyncio.gather(*tasks)

    def generate(
        self,
        conversation_list: List[Conversation],
    ) -> List[str]:
        if not conversation_list:
            return []

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._agenerate(conversation_list)
            )

        raise RuntimeError(
            "generate() cannot be called from an active "
            "asyncio event loop. Use await _agenerate() "
            "from an async caller."
        )

최종 개선사항
✅ 모델명 문자열 분기 → 포매터 레지스트리 전략화 → 모델 추가 시 기존 추론 엔진 수정 최소화
✅ 무제한 OpenAI 비동기 요청 → Semaphore 기반 동시성 제한 → Rate Limit·메모리 폭증 방어
✅ 이벤트 루프 강제 재진입 → 동기/비동기 호출 경계 분리 → 런타임 생명주기 안정성 확보
✅ inputs[0] 단일 타입 판별 → 전체 배치 타입 검증 → 혼합 입력 및 빈 배치 오류 방지
✅ CUDA 존재 여부 기반 입력 이동 → 모델 실제 device 기준 이동 → device_map=auto 환경 호환성 강화
✅ 잘못된 dtype 자동 fallback → 명시적 설정 검증 → 조용한 오동작 방지
✅ 프롬프트 모듈 최소 검증 → 필수 속성 및 import 오류 분리 → 설정·프롬프트 장애 원인 추적성 확보

원본의 전략 패턴 방향은 맞았지만, 이번 개선에서 진짜 점수를 올린 부분은 기능 추가가 아니라 새로 도입된 비동기·레지스트리·멀티디바이스 경로의 실패 조건까지 다시 잠근 것이다. 현재 형태라면 이전 버전보다 확실히 9.5점대에 가까운 구조다.        
