원본코드
import json
import os
import argparse

from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset

from module import InferenceModule, VllmModule, HfModule, OpenaiModule


BENCHMARK_IDS = ["llmbar", "hhh", "mtbench", "biasbench"]


def make_data_row(id: int, instruction: str, response1: str, response2: str, label: int) -> dict:
    return {
        "id": id,
        "instruction": instruction.strip(),
        "response1": response1.strip(),
        "response2": response2.strip(),
        "label": label,
    }


def get_benchmark_data(benchmark_id: str) -> dict:
    """output a standardized dataset. only the contents.
    the data structure will be kept until the results."""
    benchmark_set = {}
    assert benchmark_id in BENCHMARK_IDS
    print("Loading benchmark:", benchmark_id)

    if benchmark_id == "llmbar":
        dataset = load_dataset("princeton-nlp/LLMBar", trust_remote_code=True)
        for subset_name in dataset.keys():
            subset = []
            for i, row in enumerate(dataset[subset_name]):
                subset.append(make_data_row(i, row["input"], row["output_1"], row["output_2"], row["label"]))
            benchmark_set[subset_name] = subset

    elif benchmark_id == "hhh":
        for subset_name in ["helpful", "honest", "harmless", "other"]:
            subset = []
            raw_subset = load_dataset("HuggingFaceH4/hhh_alignment", name=subset_name, trust_remote_code=True)["test"]
            for i, row in enumerate(raw_subset):
                label_data = row["targets"]["labels"]
                if label_data == [1, 0]:
                    label = 1
                elif label_data == [0, 1]:
                    label = 2
                else:
                    raise ValueError(label_data)
                subset.append(make_data_row(i, row["input"], row["targets"]
                              ["choices"][0], row["targets"]["choices"][1], label))
            benchmark_set[subset_name] = subset

    elif benchmark_id == "mtbench":
        raw_dataset = load_dataset("lmsys/mt_bench_human_judgments", trust_remote_code=True)
        for subset_name in ["human", "gpt4_pair"]:
            subset = []
            for i, row in enumerate(raw_dataset[subset_name]):
                if row["turn"] == 2:
                    continue
                label_data = row["winner"]
                if label_data == "model_a":
                    label = 1
                elif label_data == "model_b":
                    label = 2
                elif label_data in ["tie", "tie (inconsistent)"]:
                    continue
                else:
                    raise ValueError(label_data)
                subset.append(make_data_row(i, row["conversation_a"][0]["content"],
                              row["conversation_a"][1]["content"], row["conversation_b"][1]["content"], label))
            benchmark_set[subset_name] = subset

    elif benchmark_id == "biasbench":
        with open("data/evalbiasbench/biasbench.json") as f:
            benchmark_set = json.load(f)
    else:
        raise ValueError(benchmark_id)

    return benchmark_set


def add_inference(benchmark_data: dict, module: InferenceModule) -> None:
    """all common logic for benchmarking. 
    apply swap, apply prompt template, apply chat template, for all subsets in benchmark data.
    run inference and update on benchmark_data"""
    conversation_list = []

    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            for swap in [False, True]:
                conversation_list.append(module.make_conversation(
                    row["instruction"], row["response1"], row["response2"], swap))

    generated_texts = module.generate(conversation_list)

    index = 0
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            result = {}
            for swap_id in ["orig", "swap"]:
                result[swap_id] = {"completion": generated_texts[index]}
                index += 1
            row["result"] = result
    assert (len(generated_texts) == index)


def add_parse_result(benchmark_data: dict, module: InferenceModule) -> None:
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            for swap, swap_id in [(False, "orig"), (True, "swap")]:
                result_dict = row["result"][swap_id]
                completion = result_dict["completion"]
                result_dict["prediction"] = module.get_prediction(completion)
                result_dict["is_correct"] = module.is_correct(result_dict["prediction"], row["label"], swap)


def get_model_statistics(run_name: str) -> dict:
    """read all inference results for the model and return scores"""
    model_stats = {}
    for benchmark_id in BENCHMARK_IDS:
        benchmark_result = {}
        filename = f"result/{run_name}/{benchmark_id}.json"
        if not os.path.exists(filename):
            print("result file", filename, "does not exist.")
            continue
        with open(filename) as f:
            data = json.load(f)
        for subset_name, subset in data.items():
            stats = {key: 0 for key in ["single_total", "single_correct", "single_accuracy",
                                        "pair_total", "pair_correct", "pair_accuracy", "pair_agree", "pair_agreement_rate"]}
            for row in subset:
                stats["single_total"] += 2
                stats["pair_total"] += 1
                if row["result"]["orig"]["is_correct"]:
                    stats["single_correct"] += 1
                if row["result"]["swap"]["is_correct"]:
                    stats["single_correct"] += 1
                if row["result"]["orig"]["is_correct"] and row["result"]["swap"]["is_correct"]:
                    stats["pair_correct"] += 1
                pred_orig = row["result"]["orig"]["prediction"]
                pred_swap = row["result"]["swap"]["prediction"]
                if set([pred_orig, pred_swap]) in [set([1, 2]), set([3])]:
                    stats["pair_agree"] += 1

            stats["single_accuracy"] = round(stats["single_correct"] / stats["single_total"]*100, 1)
            stats["pair_accuracy"] = round(stats["pair_correct"] / stats["pair_total"]*100, 1)
            stats["pair_agreement_rate"] = round(stats["pair_agree"] / stats["pair_total"]*100, 1)
            benchmark_result[subset_name] = stats
        model_stats[benchmark_id] = benchmark_result
    return model_stats


def write_model_score(run_name: str) -> None:
    """create model's score file"""
    model_stats = get_model_statistics(run_name)

    with open(f"result/{run_name}/score.json", "w") as f:
        json.dump(model_stats, fp=f, ensure_ascii=False, indent=4)


def run_benchmark(run_name: str, args: argparse.Namespace) -> None:
    """run inference, parse and score."""
    os.makedirs("result", exist_ok=True)
    os.makedirs(f"result/{run_name}", exist_ok=True)

    config = OmegaConf.load(args.config)
    OmegaConf.save(config, f"result/{run_name}/config.yaml")
    print(config)

    if (not args.hf) and (config.get("vllm_args")):
        module = VllmModule(config=config)
    elif (args.hf) and (config.get("hf_args")):
        module = HfModule(config=config)
    elif config.get("openai_args"):
        module = OpenaiModule(config=config)
    else:
        raise NotImplementedError

    for benchmark_id in args.benchmarks:
        benchmark_data = get_benchmark_data(benchmark_id)

        add_inference(benchmark_data, module)
        add_parse_result(benchmark_data, module)

        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


def run_parse(run_name: str, args: argparse.Namespace) -> None:
    """redo parsing for existing inference results, and update score."""
    config = OmegaConf.load(args.config)
    print(config)

    module = InferenceModule(config=config)
    for benchmark_id in args.benchmarks:
        with open(f"result/{run_name}/{benchmark_id}.json") as f:
            benchmark_data = json.load(f)
        add_parse_result(benchmark_data, module)
        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/offsetbias-8b.yaml")
    parser.add_argument("--name", default="", help="run name of the inference. defaults to config name.")

    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument("--benchmarks", type=list_of_strings, default=["biasbench"],
                        help="to include all benchmarks, call as '--benchmarks llmbar,hhh,mtbench,biasbench'")
    parser.add_argument("--hf", action="store_true", help="use hf generate instead of vllm")
    parser.add_argument("--parse", action="store_true", help="no inference. just parse and score.")
    parser.add_argument("--score", action="store_true", help="no inference. just score.")

    args = parser.parse_args()
    print(args)

    run_name = os.path.basename(args.config).replace(".yaml", "")
    if args.hf:
        run_name += "_hf"
    if args.name:
        run_name = args.name
    print("Run name:", run_name)

    if args.score:
        write_model_score(run_name)
    elif args.parse:
        run_parse(run_name, args)
    else:
        run_benchmark(run_name, args)

연구용 벤치마크 오케스트레이터로서는 구조가 좋지만, 가장 위험한 지점은 추론 결과를 위치 기반 인덱스로 원본 데이터에 재결합하는 구조와 결과 파일을 직접 덮어쓰는 저장 방식이 결합되어, 한 번의 부분 실패가 평가 무결성과 결과 파일 전체를 오염시킬 수 있다는 점이다.

제안패치
import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from datasets import load_dataset

from module import InferenceModule, VllmModule, HfModule, OpenaiModule


BENCHMARK_IDS = {"llmbar", "hhh", "mtbench", "biasbench"}


def make_data_row(
    row_id: int,
    instruction: str,
    response1: str,
    response2: str,
    label: int,
) -> dict:
    if not all(isinstance(value, str) for value in
               (instruction, response1, response2)):
        raise TypeError("instruction and responses must be strings.")

    if not instruction.strip() or not response1.strip() or not response2.strip():
        raise ValueError("instruction and responses must not be empty.")

    if label not in (1, 2):
        raise ValueError(f"Invalid label: {label}")

    return {
        "id": row_id,
        "instruction": instruction.strip(),
        "response1": response1.strip(),
        "response2": response2.strip(),
        "label": label,
    }


def get_benchmark_data(benchmark_id: str) -> dict:
    if benchmark_id not in BENCHMARK_IDS:
        raise ValueError(
            f"Unsupported benchmark: {benchmark_id}. "
            f"Expected one of {sorted(BENCHMARK_IDS)}."
        )

    benchmark_set = {}

    if benchmark_id == "llmbar":
        dataset = load_dataset(
            "princeton-nlp/LLMBar",
            trust_remote_code=True,
        )

        for subset_name in dataset.keys():
            subset = []
            for i, row in enumerate(dataset[subset_name]):
                subset.append(
                    make_data_row(
                        i,
                        row["input"],
                        row["output_1"],
                        row["output_2"],
                        row["label"],
                    )
                )
            benchmark_set[subset_name] = subset

    elif benchmark_id == "hhh":
        for subset_name in ["helpful", "honest", "harmless", "other"]:
            raw_subset = load_dataset(
                "HuggingFaceH4/hhh_alignment",
                name=subset_name,
                trust_remote_code=True,
            )["test"]

            subset = []

            for i, row in enumerate(raw_subset):
                label_data = row["targets"]["labels"]

                if label_data == [1, 0]:
                    label = 1
                elif label_data == [0, 1]:
                    label = 2
                else:
                    raise ValueError(
                        f"Unsupported HHH label format: {label_data}"
                    )

                subset.append(
                    make_data_row(
                        i,
                        row["input"],
                        row["targets"]["choices"][0],
                        row["targets"]["choices"][1],
                        label,
                    )
                )

            benchmark_set[subset_name] = subset

    elif benchmark_id == "mtbench":
        raw_dataset = load_dataset(
            "lmsys/mt_bench_human_judgments",
            trust_remote_code=True,
        )

        for subset_name in ["human", "gpt4_pair"]:
            subset = []

            for i, row in enumerate(raw_dataset[subset_name]):
                if row["turn"] == 2:
                    continue

                label_data = row["winner"]

                if label_data == "model_a":
                    label = 1
                elif label_data == "model_b":
                    label = 2
                elif label_data in {"tie", "tie (inconsistent)"}:
                    continue
                else:
                    raise ValueError(
                        f"Unsupported MT-Bench winner: {label_data}"
                    )

                subset.append(
                    make_data_row(
                        i,
                        row["conversation_a"][0]["content"],
                        row["conversation_a"][1]["content"],
                        row["conversation_b"][1]["content"],
                        label,
                    )
                )

            benchmark_set[subset_name] = subset

    elif benchmark_id == "biasbench":
        path = Path("data/evalbiasbench/biasbench.json")

        if not path.is_file():
            raise FileNotFoundError(
                f"BiasBench dataset file not found: {path}"
            )

        with path.open(encoding="utf-8") as f:
            benchmark_set = json.load(f)

    if not benchmark_set:
        raise RuntimeError(
            f"Benchmark {benchmark_id} produced no subsets."
        )

    return benchmark_set


def add_inference(
    benchmark_data: dict,
    module: InferenceModule,
) -> None:
    conversations = []
    expected_items = []

    for subset_name, subset_data in benchmark_data.items():
        for row_index, row in enumerate(subset_data):
            for swap, swap_id in ((False, "orig"), (True, "swap")):
                conversations.append(
                    module.make_conversation(
                        row["instruction"],
                        row["response1"],
                        row["response2"],
                        swap,
                    )
                )
                expected_items.append(
                    (subset_name, row_index, swap_id)
                )

    if not conversations:
        raise RuntimeError("No conversations were generated.")

    generated_texts = module.generate(conversations)

    if not isinstance(generated_texts, list):
        raise TypeError(
            "Inference backend must return a list of generated texts."
        )

    if len(generated_texts) != len(expected_items):
        raise RuntimeError(
            "Inference result count mismatch: "
            f"expected {len(expected_items)}, "
            f"got {len(generated_texts)}."
        )

    for generated_text, (
        subset_name,
        row_index,
        swap_id,
    ) in zip(generated_texts, expected_items):
        if not isinstance(generated_text, str):
            raise TypeError(
                f"Generated result must be str, "
                f"got {type(generated_text).__name__}."
            )

        benchmark_data[subset_name][row_index].setdefault(
            "result", {}
        )[swap_id] = {
            "completion": generated_text
        }


def add_parse_result(
    benchmark_data: dict,
    module: InferenceModule,
) -> None:
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            result = row.get("result")

            if not isinstance(result, dict):
                raise ValueError(
                    f"Missing result data for row {row.get('id')} "
                    f"in subset {subset_name}."
                )

            for swap, swap_id in ((False, "orig"), (True, "swap")):
                result_dict = result.get(swap_id)

                if not isinstance(result_dict, dict):
                    raise ValueError(
                        f"Missing {swap_id} result for row {row.get('id')}."
                    )

                completion = result_dict.get("completion")

                if not isinstance(completion, str) or not completion.strip():
                    raise ValueError(
                        f"Empty completion for row {row.get('id')} "
                        f"({swap_id})."
                    )

                prediction = module.get_prediction(completion)

                result_dict["prediction"] = prediction
                result_dict["is_correct"] = module.is_correct(
                    prediction,
                    row["label"],
                    swap,
                )


def get_model_statistics(run_name: str) -> dict:
    model_stats = {}

    for benchmark_id in BENCHMARK_IDS:
        benchmark_result = {}
        filename = Path("result") / run_name / f"{benchmark_id}.json"

        if not filename.is_file():
            continue

        with filename.open(encoding="utf-8") as f:
            data = json.load(f)

        for subset_name, subset in data.items():
            stats = {
                "single_total": 0,
                "single_correct": 0,
                "single_accuracy": 0.0,
                "pair_total": 0,
                "pair_correct": 0,
                "pair_accuracy": 0.0,
                "pair_agree": 0,
                "pair_agreement_rate": 0.0,
            }

            for row in subset:
                orig = row["result"]["orig"]
                swap = row["result"]["swap"]

                stats["single_total"] += 2
                stats["pair_total"] += 1

                if orig["is_correct"]:
                    stats["single_correct"] += 1

                if swap["is_correct"]:
                    stats["single_correct"] += 1

                if orig["is_correct"] and swap["is_correct"]:
                    stats["pair_correct"] += 1

                predictions = {orig["prediction"], swap["prediction"]}

                if predictions in ({1, 2}, {3}):
                    stats["pair_agree"] += 1

            if stats["single_total"] > 0:
                stats["single_accuracy"] = round(
                    stats["single_correct"]
                    / stats["single_total"]
                    * 100,
                    1,
                )

            if stats["pair_total"] > 0:
                stats["pair_accuracy"] = round(
                    stats["pair_correct"]
                    / stats["pair_total"]
                    * 100,
                    1,
                )
                stats["pair_agreement_rate"] = round(
                    stats["pair_agree"]
                    / stats["pair_total"]
                    * 100,
                    1,
                )

            benchmark_result[subset_name] = stats

        model_stats[benchmark_id] = benchmark_result

    return model_stats


def _atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.flush()

        temp_path.replace(path)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_model_score(run_name: str) -> None:
    model_stats = get_model_statistics(run_name)
    output_path = Path("result") / run_name / "score.json"
    _atomic_json_dump(model_stats, output_path)


def run_benchmark(
    run_name: str,
    args: argparse.Namespace,
) -> None:
    result_dir = Path("result") / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    OmegaConf.save(config, result_dir / "config.yaml")

    if not args.hf and config.get("vllm_args"):
        module = VllmModule(config=config)
    elif args.hf and config.get("hf_args"):
        module = HfModule(config=config)
    elif config.get("openai_args"):
        module = OpenaiModule(config=config)
    else:
        raise NotImplementedError(
            "No supported inference backend configuration found."
        )

    for benchmark_id in args.benchmarks:
        benchmark_data = get_benchmark_data(benchmark_id)

        add_inference(benchmark_data, module)
        add_parse_result(benchmark_data, module)

        output_path = result_dir / f"{benchmark_id}.json"
        _atomic_json_dump(benchmark_data, output_path)

    write_model_score(run_name)


def run_parse(
    run_name: str,
    args: argparse.Namespace,
) -> None:
    config = OmegaConf.load(args.config)
    module = InferenceModule(config=config)

    for benchmark_id in args.benchmarks:
        path = Path("result") / run_name / f"{benchmark_id}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"Benchmark result not found: {path}"
            )

        with path.open(encoding="utf-8") as f:
            benchmark_data = json.load(f)

        add_parse_result(benchmark_data, module)
        _atomic_json_dump(benchmark_data, path)

    write_model_score(run_name)


def list_of_strings(value: str) -> list[str]:
    benchmarks = [item.strip() for item in value.split(",") if item.strip()]

    if not benchmarks:
        raise argparse.ArgumentTypeError(
            "At least one benchmark must be specified."
        )

    invalid = set(benchmarks) - BENCHMARK_IDS

    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported benchmarks: {sorted(invalid)}"
        )

    return benchmarks


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/offsetbias-8b.yaml",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Run name. Defaults to config filename.",
    )
    parser.add_argument(
        "--benchmarks",
        type=list_of_strings,
        default=["biasbench"],
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Use HuggingFace generation instead of vLLM.",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Skip inference and only parse results.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Skip inference and parsing and only calculate scores.",
    )

    args = parser.parse_args()

    run_name = Path(args.config).stem

    if args.hf:
        run_name += "_hf"

    if args.name:
        run_name = args.name

    if args.score:
        write_model_score(run_name)
    elif args.parse:
        run_parse(run_name, args)
    else:
        run_benchmark(run_name, args)


if __name__ == "__main__":
    main()

최종 개선사항
✅ 결과 개수와 원본 데이터의 순서만 묵시적으로 신뢰 → 요청·응답 개수 계약 검증 → 추론 결과의 잘못된 샘플 매핑 및 평가 오염 방지
✅ index 기반 결과 재결합 → 명시적인 subset·row·swap 매핑 유지 → A/B 평가 결과의 위치 무결성 확보
✅ 빈 subset에서 무조건 정확도 계산 → 분모 존재 여부 검증 후 계산 → 데이터 필터링으로 인한 ZeroDivisionError 차단
✅ 로컬 파일을 바로 open() → 경로 존재 여부와 인코딩을 명시적으로 검증 → 데이터셋 누락 시 원인 추적 가능한 실패 보장
✅ 결과 JSON을 기존 파일에 직접 덮어쓰기 → 임시 파일 완성 후 원자적 교체 → 추론 중단이나 저장 실패에 의한 기존 결과 손상 방지
✅ assert와 묵시적 자료형 계약 의존 → 런타임 입력·출력 타입 및 결과 구조 검증 → 최적화 실행에서도 유지되는 장애 방어선 확보
✅ 전체 추론 배치를 무검증으로 전달 → 생성 결과 타입·개수·completion 유효성 검증 → 비정상 모델 출력을 정상 평가 데이터로 저장하는 문제 차단

대규모 벤치마크의 순차 매핑·직접 덮어쓰기·분모 미검증이라는 핵심 약점을 제거해, 평가 파이프라인의 실행 안정성과 결과 무결성을 동시에 확보한 실무형 구조로 승격했다.
