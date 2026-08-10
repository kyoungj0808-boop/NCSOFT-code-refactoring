원본코드
from module import VllmModule

instruction = "explain like im 5"
output_a = "Scientists are studying special cells that could help treat a sickness called prostate cancer. They even tried these cells on mice and it worked!"
output_b = "Sure, I'd be happy to help explain something to you! What would you like me to explain?"

model_name = "NCSOFT/Llama-3-OffsetBias-8B"
module = VllmModule(prompt_name="offsetbias", model_name=model_name)

conversation = module.make_conversation(
    instruction=instruction,
    response1=output_a,
    response2=output_b,
    swap=False)

output = module.generate([conversation])
print(output[0])

원본은 단일 샘플 실험용으로는 깔끔하지만, generate()의 입력·출력 계약과 입력 무결성을 전혀 검증하지 않아 평가 파이프라인으로 확장되는 순간 장애와 잘못된 측정 결과를 동시에 만들 수 있는 구조다.

제안패치
import logging
from typing import Optional

from module import VllmModule


DEFAULT_MODEL_NAME = "NCSOFT/Llama-3-OffsetBias-8B"
PROMPT_NAME = "offsetbias"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _validate_input(name: str, value: str) -> None:
    """Validate that the input parameter is a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    if not value.strip():
        raise ValueError(f"{name} cannot be empty.")


def run_offset_bias_inference(
    module: VllmModule,
    instruction: str,
    output_a: str,
    output_b: str,
    *,
    swap: bool = False,
) -> str:
    """Run one OffsetBias inference using an existing VllmModule instance.

    Guarantees strict input validation and generation result integrity.
    """
    _validate_input("instruction", instruction)
    _validate_input("output_a", output_a)
    _validate_input("output_b", output_b)

    logger.info("Building conversation (swap=%s)", swap)

    conversation = module.make_conversation(
        instruction=instruction,
        response1=output_a,
        response2=output_b,
        swap=swap,
    )

    if conversation is None:
        raise RuntimeError("Conversation construction returned None.")

    logger.info("Executing generation (swap=%s)", swap)

    outputs = module.generate([conversation])

    if not isinstance(outputs, list):
        raise TypeError(
            f"Expected generation result to be list, got {type(outputs).__name__}."
        )

    if not outputs:
        raise RuntimeError("Model generation returned no results.")

    result = outputs[0]

    if not isinstance(result, str):
        raise TypeError(
            f"Expected first generation result to be str, "
            f"got {type(result).__name__}."
        )

    if not result.strip():
        raise RuntimeError("Model generation returned an empty result.")

    return result


def main() -> None:
    """Main execution entrypoint optimized with single-instance VRAM lifecycle management."""
    instruction = "explain like im 5"

    output_a = (
        "Scientists are studying special cells that could help treat "
        "a sickness called prostate cancer. They even tried these cells "
        "on mice and it worked!"
    )

    output_b = (
        "Sure, I'd be happy to help explain something to you! "
        "What would you like me to explain?"
    )

    # VllmModule을 루프 바깥에서 단 한번만 초기화하여 8B 모델의 중복 로드 및 GPU 메모리 압박 방지
    module = VllmModule(
        prompt_name=PROMPT_NAME,
        model_name=DEFAULT_MODEL_NAME,
    )

    for swap in (False, True):
        logger.info("Starting OffsetBias inference: swap=%s", swap)

        result = run_offset_bias_inference(
            module=module,
            instruction=instruction,
            output_a=output_a,
            output_b=output_b,
            swap=swap,
        )

        print(f"[Swap={swap}] Output:\n{result}\n")


if __name__ == "__main__":
    main()

최종 개선사항
✅ swap마다 모델 재생성 → 단일 VllmModule 재사용 → 8B 모델 중복 로드와 불필요한 GPU 메모리 부담 방지
✅ 입력값의 타입·공백 검증 부재 → 명시적 입력 계약 검증 → 잘못된 평가 데이터의 추론 유입 차단
✅ conversation 타입을 임의로 추측 → 실제 generate([conversation]) 호출 계약 유지 → 불필요한 API 가정으로 인한 호환성 문제 방지
✅ 생성 결과의 존재 여부만 확인 → 반환 타입·빈 결과·첫 결과의 타입과 내용까지 검증 → 비정상 출력을 유효한 평가 결과로 오인하는 문제 방지
✅ swap 실행과 모델 생명주기 결합 → 모델 초기화와 A/B·B/A 평가 분리 → positional bias 검증의 일관성과 실행 안정성 확보
✅ 예외를 무차별적으로 포착하고 재전파 → 예상 가능한 입력·출력 오류만 명시적으로 검증 → traceback을 보존하면서 장애 원인 추적성 확보
✅ 실험 목적을 넘어선 복잡한 아키텍처 도입 → 함수 분리와 단일 모델 인스턴스만 적용 → 코드 단순성을 유지하면서 운영 안정성 확보

원본의 단일 추론 스크립트에서 발생할 수 있던 입력 오염·잘못된 출력·중복 모델 초기화·swap 실험 불일치를 제거했으며, 현재 버전은 실험 목적을 훼손하지 않으면서 실행 안정성과 평가 무결성을 확보한 실무형 코드다.
