from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import os

import yaml

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - allows running without torch installed
    torch = None

from eagle.evaluation import eval_eagle

from .scorers import get_scorer


DEFAULT_CONFIG_PATH = Path("configs/model_suites.yaml")

DEFAULT_QUESTION_FILES = {
    "gsm8k": Path("eagle/data/gsm8k/question.jsonl"),
    "sum": Path("eagle/data/sum/question.jsonl"),
    "mt_bench": Path("eagle/data/mt_bench/question.jsonl"),
    "alpaca": Path("eagle/data/alpaca/question.jsonl"),
    "qa": Path("eagle/data/qa/question.jsonl"),
    "humaneval": Path("eagle/data/humaneval/question.jsonl"),
}


@dataclass
class ModelSpec:
    id: str
    base_model: str
    ea_model: str
    max_active_experts: List[int]
    cuda_devices: Optional[str]


@dataclass
class RunSpec:
    model_spec: ModelSpec
    variant: str
    use_eagle: bool
    max_active_experts: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EAGLE evaluation suites")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML file describing model suites",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated benchmark names (e.g., gsm8k,sum)",
    )
    parser.add_argument(
        "--dataset-file",
        action="append",
        default=[],
        help="Override question file for a dataset (format: bench=path)",
    )
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--total-token", type=int, default=63, help="Draft tokens for EAGLE")
    parser.add_argument("--warmup-tokens", type=int, default=16, help="Warmup tokens before evaluation")
    parser.add_argument("--run-name", type=str, default=None, help="Optional tag for this sweep")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate answers even if they exist")
    parser.add_argument("--default-gpus", type=str, default=None, help="Fallback CUDA_VISIBLE_DEVICES to use when a model does not specify GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    return parser.parse_args()


def load_model_specs(path: Path) -> List[ModelSpec]:
    data = yaml.safe_load(path.read_text())
    models = data.get("models", [])
    specs: List[ModelSpec] = []
    for entry in models:
        specs.append(
            ModelSpec(
                id=entry["id"],
                base_model=entry["base_model"],
                ea_model=entry["ea_model"],
                max_active_experts=list(entry.get("max_active_experts", [])),
                cuda_devices=entry.get("gpus"),
            )
        )
    return specs


def expand_runs(spec: ModelSpec) -> Iterable[RunSpec]:
    yield RunSpec(spec, variant=f"{spec.id}-baseline", use_eagle=False, max_active_experts=None)
    yield RunSpec(spec, variant=f"{spec.id}-eagle", use_eagle=True, max_active_experts=None)

    for budget in spec.max_active_experts:
        suffix = f"B{budget}"
        budget_value = None if budget == 0 else budget
        yield RunSpec(
            spec,
            variant=f"{spec.id}-eagle_{suffix}",
            use_eagle=True,
            max_active_experts=budget_value,
        )


def resolve_datasets(dataset_arg: str, overrides: List[str]) -> Dict[str, Path]:
    datasets = [item.strip() for item in dataset_arg.split(",") if item.strip()]
    if not datasets:
        raise ValueError("No datasets specified")

    override_map: Dict[str, Path] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-file entry: {item}")
        bench, path = item.split("=", 1)
        override_map[bench.strip()] = Path(path.strip())

    result: Dict[str, Path] = {}
    for bench in datasets:
        if bench in override_map:
            result[bench] = override_map[bench]
        elif bench in DEFAULT_QUESTION_FILES:
            result[bench] = DEFAULT_QUESTION_FILES[bench]
        else:
            raise ValueError(f"No question file known for '{bench}'. Use --dataset-file to provide one.")
    return result


def default_answer_file(bench: str, variant: str, mode: str, dataset_tag: str, temperature: float) -> Path:
    filename = f"{variant}-{mode}-{dataset_tag}-t{temperature}".rstrip("-") + ".jsonl"
    return Path("results") / bench / filename


def default_metrics_file(bench: str, variant: str, dataset_tag: str) -> Path:
    filename = f"{variant}-{dataset_tag}-metrics.json"
    return Path("results") / bench / filename


def build_args_namespace(args: argparse.Namespace, bench: str, question_file: Path, use_eagle: bool, use_eagle3: bool, max_active_experts: Optional[int]) -> SimpleNamespace:
    return SimpleNamespace(
        bench_name=bench,
        question_file=str(question_file),
        question_begin=None,
        question_end=None,
        num_questions=args.num_questions,
        max_new_tokens=args.max_new_tokens,
        warmup_tokens=args.warmup_tokens,
        temperature=args.temperature,
        use_eagle=use_eagle,
        use_eagle3=use_eagle3,
        total_token=args.total_token,
        max_active_experts=max_active_experts,
        collect_expert_traces=False,
        answer_file=None,
        ea_model_path=None,
        base_model_path=None,
        conv_template="vicuna",
    )


@contextmanager
def temporary_cuda_visible(devices: Optional[str]):
    if devices is None:
        yield
        return

    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        yield
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2)
    return float(sorted_vals[mid])


def summarize_generation_stats(all_stats: Dict[str, List[Any]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not all_stats:
        return summary

    def mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    throughput = all_stats.get("tokens_per_second", [])
    summary["mean_throughput"] = mean(throughput)
    summary["median_throughput"] = median(throughput)
    if "tokens_per_iter" in all_stats:
        summary["mean_tokens_per_iter"] = mean(all_stats.get("tokens_per_iter", []))
    if "avg_accept_length" in all_stats:
        summary["mean_accept_length"] = mean(all_stats.get("avg_accept_length", []))
    return summary


def flatten_numeric_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}{key}"
        if isinstance(value, (int, float)):
            result[full_key] = float(value)
        elif isinstance(value, dict):
            nested = flatten_numeric_metrics(value, prefix=f"{full_key}_")
            result.update(nested)
    return result


def main() -> None:
    args = parse_args()
    model_specs = load_model_specs(args.config)
    bench_to_question = resolve_datasets(args.datasets, args.dataset_file)

    run_name = args.run_name or Path(args.config).stem
    summary_rows: List[Dict[str, Any]] = []

    for bench, question_file in bench_to_question.items():
        question_path = Path(question_file)
        dataset_tag = question_path.stem

        for spec in model_specs:
            for run in expand_runs(spec):
                mode = "eagle" if run.use_eagle else "baseline"
                answer_path = default_answer_file(bench, run.variant, mode, dataset_tag, args.temperature)
                metrics_path = default_metrics_file(bench, run.variant, dataset_tag)

                answer_exists = answer_path.exists()
                generation_stats: Dict[str, float] = {}

                if args.dry_run:
                    print(
                        f"[DRY] {bench} :: {run.variant} -> {answer_path}"
                        f" (CUDA_VISIBLE_DEVICES={run.model_spec.cuda_devices or args.default_gpus or os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
                    )
                    continue

                if answer_exists and not args.overwrite:
                    print(f"Skipping generation for {run.variant} ({bench}); answer exists")
                else:
                    answer_path.parent.mkdir(parents=True, exist_ok=True)

                    ns = build_args_namespace(
                        args,
                        bench,
                        question_path,
                        run.use_eagle,
                        use_eagle3=True,
                        max_active_experts=run.max_active_experts,
                    )

                    devices = run.model_spec.cuda_devices or args.default_gpus
                    print(
                        f"Running {bench} :: {run.variant}"
                        + (f" [CUDA_VISIBLE_DEVICES={devices}]" if devices else "")
                    )
                    try:
                        with temporary_cuda_visible(devices):
                            all_stats = eval_eagle.run_evaluation(
                                base_model_path=run.model_spec.base_model,
                                ea_model_path=run.model_spec.ea_model,
                                model_id=run.variant,
                                conv_template="vicuna",
                                question_file=str(question_path),
                                answer_file=str(answer_path),
                                num_questions=args.num_questions,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                use_eagle=run.use_eagle,
                                args=ns,
                            )
                        generation_stats = summarize_generation_stats(all_stats)
                    except Exception as exc:
                        print(f"[error] Generation failed for {bench} :: {run.variant} — {exc}")
                        continue
                    finally:
                        if torch is not None and torch.cuda.is_available():
                            for device_index in range(torch.cuda.device_count()):
                                with torch.cuda.device(device_index):
                                    torch.cuda.empty_cache()

                if metrics_path.exists() and not args.overwrite and answer_exists:
                    with metrics_path.open() as fh:
                        metrics = json.load(fh)
                        numeric_metrics = flatten_numeric_metrics(metrics)
                        summary_rows.append(
                            {
                                "bench": bench,
                                "model_variant": run.variant,
                                "dataset": dataset_tag,
                                **numeric_metrics,
                            }
                        )
                    continue

                scorer = get_scorer(bench)
                if scorer is None:
                    print(f"No scorer registered for {bench}; skipping metrics")
                    continue

                run_metadata = {
                    "bench_name": bench,
                    "model_id": run.variant,
                    "variant": run.variant,
                    "generation_stats": generation_stats,
                    "config": {
                        "base_model_id": spec.id,
                        "base_model": run.model_spec.base_model,
                        "ea_model": run.model_spec.ea_model,
                        "use_eagle": run.use_eagle,
                        "max_active_experts": run.max_active_experts,
                        "num_questions": args.num_questions,
                        "temperature": args.temperature,
                        "total_token": args.total_token,
                    },
                }

                metrics = scorer(
                    answer_file=answer_path,
                    question_file=question_path,
                    output_path=metrics_path,
                    run_metadata=run_metadata,
                )

                numeric_metrics = flatten_numeric_metrics(metrics)
                printable = {
                    key: value
                    for key, value in numeric_metrics.items()
                    if not key.startswith("details_")
                }
                if printable:
                    pretty = ", ".join(f"{k}={v:.4f}" for k, v in sorted(printable.items()))
                    print(f"Scoring complete: {bench} :: {run.variant} ({pretty})")

                summary_rows.append(
                    {
                        "bench": bench,
                        "model_variant": run.variant,
                        "dataset": dataset_tag,
                        **numeric_metrics,
                    }
                )

    if summary_rows and not args.dry_run:
        summary_dir = Path("results") / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{run_name}.json"
        with summary_path.open("w") as fh:
            json.dump(summary_rows, fh, indent=2)
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
