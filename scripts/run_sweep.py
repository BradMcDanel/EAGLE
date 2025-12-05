#!/usr/bin/env python3
"""Run parameter sweeps for EAGLE evaluation experiments.

Usage:
    python scripts/run_sweep.py sweeps/humaneval_test.yaml --gpus 0,1,2,3
    python scripts/run_sweep.py sweeps/humaneval_test.yaml --gpus 0 --dry-run
    python scripts/run_sweep.py sweeps/humaneval_test.yaml --gpus 0,1 --overwrite
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add repo root to path
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@dataclass
class Task:
    """Represents a single experiment run."""
    dataset: str
    temperature: float
    mode: str  # "autoregressive", "eagle_uncapped", or "eagle_pruned"
    draft_tokens: Optional[int]
    pruning_strategy: Optional[str]
    selection_mode: Optional[str]  # "cardinality" or "probability"
    budget: Optional[float]  # int for cardinality, float for probability
    model_base_path: str
    model_eagle_path: str
    num_samples: Optional[int]

    def get_model_id(self) -> str:
        """Generate model ID based on task parameters."""
        from eagle.evaluation.eval_eagle import get_model_id

        if self.mode == "autoregressive":
            return get_model_id(
                base_model_path=self.model_base_path,
                use_eagle=False,
                total_token=0,
                expert_selection_mode="cardinality",
                expert_count_budget=None,
                expert_probability_budget=None,
                expert_pruning_strategy="substitution",
                temperature=self.temperature,
            )
        elif self.mode == "eagle_uncapped":
            return get_model_id(
                base_model_path=self.model_base_path,
                use_eagle=True,
                total_token=self.draft_tokens,
                expert_selection_mode="cardinality",
                expert_count_budget=None,
                expert_probability_budget=None,
                expert_pruning_strategy="substitution",
                temperature=self.temperature,
            )
        else:  # eagle_pruned
            budget_int = int(self.budget) if self.selection_mode == "cardinality" else None
            budget_float = float(self.budget) if self.selection_mode == "probability" else None
            return get_model_id(
                base_model_path=self.model_base_path,
                use_eagle=True,
                total_token=self.draft_tokens,
                expert_selection_mode=self.selection_mode,
                expert_count_budget=budget_int,
                expert_probability_budget=budget_float,
                expert_pruning_strategy=self.pruning_strategy,
                temperature=self.temperature,
            )

    def get_output_dir(self) -> Path:
        """Get the output directory for this task."""
        model_id = self.get_model_id()
        return Path("results") / self.dataset / model_id

    def build_command(self) -> List[str]:
        """Build the command to execute this task."""
        cmd = [
            sys.executable,
            "eagle/evaluation/eval_eagle.py",
            "--base-model-path", self.model_base_path,
            "--ea-model-path", self.model_eagle_path,
            "--bench-name", self.dataset,
            "--temperature", str(self.temperature),
        ]

        # Add num-samples if specified
        if self.num_samples is not None:
            cmd.extend(["--num-questions", str(self.num_samples)])

        # Mode-specific arguments
        if self.mode == "autoregressive":
            pass
        elif self.mode == "eagle_uncapped":
            cmd.extend([
                "--use-eagle",
                "--total-token", str(self.draft_tokens),
            ])
        else:  # eagle_pruned
            cmd.extend([
                "--use-eagle",
                "--total-token", str(self.draft_tokens),
                "--expert-selection-mode", self.selection_mode,
                "--expert-pruning-strategy", self.pruning_strategy,
            ])

            if self.selection_mode == "cardinality":
                cmd.extend(["--expert-count-budget", str(int(self.budget))])
            else:  # probability
                cmd.extend(["--expert-probability-budget", str(self.budget)])

        return cmd

    def __str__(self) -> str:
        """Human-readable task description."""
        if self.mode == "autoregressive":
            return f"{self.dataset} | temp={self.temperature} | autoregressive"
        elif self.mode == "eagle_uncapped":
            return f"{self.dataset} | temp={self.temperature} | eagle_uncapped | draft={self.draft_tokens}"
        else:
            budget_str = f"{int(self.budget)}" if self.selection_mode == "cardinality" else f"{self.budget:.2f}"
            return (f"{self.dataset} | temp={self.temperature} | draft={self.draft_tokens} | "
                   f"{self.selection_mode}={budget_str} | {self.pruning_strategy}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate sweep configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    assert "model" in config, "Config must have 'model' section"
    assert "base_path" in config["model"], "model.base_path required"
    assert "eagle_path" in config["model"], "model.eagle_path required"
    assert "gpus_per_run" in config["model"], "model.gpus_per_run required"
    assert "datasets" in config, "datasets required"
    assert "temperatures" in config, "temperatures required"
    assert "draft_tokens" in config, "draft_tokens required"
    assert "pruning_strategies" in config, "pruning_strategies required"
    assert "expert_selection" in config, "expert_selection required"

    return config


def generate_tasks(config: Dict[str, Any]) -> List[Task]:
    """Generate all tasks from config."""
    tasks = []

    model_base = config["model"]["base_path"]
    model_eagle = config["model"]["eagle_path"]
    datasets = config["datasets"]
    temperatures = config["temperatures"]
    num_samples = config.get("num_samples")

    # Generate baseline tasks (autoregressive + eagle_uncapped)
    for dataset in datasets:
        for temp in temperatures:
            # Autoregressive baseline
            tasks.append(Task(
                dataset=dataset,
                temperature=temp,
                mode="autoregressive",
                draft_tokens=None,
                pruning_strategy=None,
                selection_mode=None,
                budget=None,
                model_base_path=model_base,
                model_eagle_path=model_eagle,
                num_samples=num_samples,
            ))

            # Eagle uncapped baseline for each draft token setting
            for draft_token in config["draft_tokens"]:
                tasks.append(Task(
                    dataset=dataset,
                    temperature=temp,
                    mode="eagle_uncapped",
                    draft_tokens=draft_token,
                    pruning_strategy=None,
                    selection_mode=None,
                    budget=None,
                    model_base_path=model_base,
                    model_eagle_path=model_eagle,
                    num_samples=num_samples,
                ))

    # Generate experiment tasks (cross product)
    draft_tokens = config["draft_tokens"]
    pruning_strategies = config["pruning_strategies"]
    expert_selection = config["expert_selection"]

    # Cardinality budgets
    cardinality_budgets = expert_selection.get("cardinality", [])
    for dataset, temp, draft, strategy, budget in product(
        datasets, temperatures, draft_tokens, pruning_strategies, cardinality_budgets
    ):
        tasks.append(Task(
            dataset=dataset,
            temperature=temp,
            mode="eagle_pruned",
            draft_tokens=draft,
            pruning_strategy=strategy,
            selection_mode="cardinality",
            budget=budget,
            model_base_path=model_base,
            model_eagle_path=model_eagle,
            num_samples=num_samples,
        ))

    # Probability budgets
    probability_budgets = expert_selection.get("probability", [])
    for dataset, temp, draft, strategy, budget in product(
        datasets, temperatures, draft_tokens, pruning_strategies, probability_budgets
    ):
        tasks.append(Task(
            dataset=dataset,
            temperature=temp,
            mode="eagle_pruned",
            draft_tokens=draft,
            pruning_strategy=strategy,
            selection_mode="probability",
            budget=budget,
            model_base_path=model_base,
            model_eagle_path=model_eagle,
            num_samples=num_samples,
        ))

    return tasks


def task_exists(task: Task) -> bool:
    """Check if task output already exists."""
    output_dir = task.get_output_dir()
    summary_file = output_dir / "summary.json"
    return summary_file.exists()


def run_task(task: Task, gpu_queue) -> Dict[str, Any]:
    """Execute a single task on assigned GPUs."""
    # Get GPUs when task actually starts (blocks until GPUs available)
    gpu_ids = gpu_queue.get()

    try:
        # Set CUDA_VISIBLE_DEVICES
        gpu_str = ",".join(str(g) for g in gpu_ids)
        env = {"CUDA_VISIBLE_DEVICES": gpu_str}

        # Build command
        cmd = task.build_command()
        result = subprocess.run(
            cmd,
            env={**subprocess.os.environ, **env},
            capture_output=True,
            text=True,
            check=False,
        )

        success = result.returncode == 0

        if not success:
            return {
                "task": task,
                "task_str": str(task),
                "success": False,
                "returncode": result.returncode,
                "error": result.stderr[:500] if result.stderr else "Unknown error",
            }

        # Parse metrics if successful
        metrics = {}
        summary = {}
        output_dir = task.get_output_dir()
        metrics_file = output_dir / "metrics.json"
        summary_file = output_dir / "summary.json"

        if metrics_file.exists():
            import json
            with open(metrics_file) as f:
                metrics_data = json.load(f)
                # Extract key metric based on benchmark
                if "pass@1" in metrics_data:
                    metrics["pass@1"] = metrics_data["pass@1"]
                elif "accuracy" in metrics_data:
                    metrics["accuracy"] = metrics_data["accuracy"]
                elif "rouge_l" in metrics_data:
                    metrics["rouge_l"] = metrics_data["rouge_l"].get("f1", 0.0)
                elif "f1" in metrics_data:
                    metrics["f1"] = metrics_data["f1"]

        if summary_file.exists():
            import json
            with open(summary_file) as f:
                summary_data = json.load(f)
                summary["throughput"] = summary_data.get("mean_throughput", 0.0)
                summary["accept_length"] = summary_data.get("mean_accept_length", 0.0)

        return {
            "task": task,
            "task_str": str(task),
            "success": True,
            "metrics": metrics,
            "summary": summary,
        }

    except Exception as e:
        return {
            "task": task,
            "task_str": str(task),
            "success": False,
            "error": str(e),
        }
    finally:
        # Always return GPUs to queue
        gpu_queue.put(gpu_ids)


def main():
    parser = argparse.ArgumentParser(description="Run EAGLE parameter sweep")
    parser.add_argument("config", type=Path, help="Path to sweep config YAML file")
    parser.add_argument("--gpus", type=str, required=True,
                       help="Comma-separated list of GPU IDs (e.g., 0,1,2,3)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print tasks without executing")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing results (default: skip)")

    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Override overwrite setting if flag provided
    if args.overwrite:
        config["overwrite"] = True

    # Parse GPU list
    gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
    gpus_per_run = config["model"]["gpus_per_run"]
    num_workers = len(gpu_list) // gpus_per_run

    assert len(gpu_list) % gpus_per_run == 0, \
        f"GPU count ({len(gpu_list)}) must be divisible by gpus_per_run ({gpus_per_run})"

    print(f"GPUs: {gpu_list} | Workers: {num_workers} | GPUs per run: {gpus_per_run}")

    # Generate tasks
    print("\nGenerating tasks...")
    all_tasks = generate_tasks(config)
    print(f"Total tasks: {len(all_tasks)}")

    # Filter existing (unless overwrite)
    if config.get("overwrite", False):
        pending_tasks = all_tasks
        print("Overwrite mode: will re-run all tasks")
    else:
        pending_tasks = [t for t in all_tasks if not task_exists(t)]
        completed = len(all_tasks) - len(pending_tasks)
        print(f"Completed: {completed} | Pending: {len(pending_tasks)}")

    if args.dry_run:
        print("\n=== DRY RUN: Tasks to execute ===")
        for i, task in enumerate(pending_tasks, 1):
            print(f"{i:3d}. {task}")
        return

    if not pending_tasks:
        print("\nNo pending tasks. All experiments completed!")
        return

    # Execute tasks in parallel with dynamic GPU allocation
    print(f"\nExecuting {len(pending_tasks)} tasks with {num_workers} workers...")

    # Create GPU queue using Manager for cross-process sharing
    with Manager() as manager:
        gpu_queue = manager.Queue()
        for worker_id in range(num_workers):
            start_gpu_idx = worker_id * gpus_per_run
            worker_gpus = gpu_list[start_gpu_idx:start_gpu_idx + gpus_per_run]
            gpu_queue.put(worker_gpus)

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks - each will grab GPUs when it starts
            futures = {executor.submit(run_task, task, gpu_queue): task for task in pending_tasks}

            # Wait for completion
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "✓" if result["success"] else "✗"
                progress = f"[{len(results)}/{len(pending_tasks)}]"

                if result["success"]:
                    # Format metrics for display
                    metrics_str = ""
                    if result.get("metrics"):
                        metrics = result["metrics"]
                        if "pass@1" in metrics:
                            metrics_str = f" | pass@1={metrics['pass@1']:.3f}"
                        elif "accuracy" in metrics:
                            metrics_str = f" | acc={metrics['accuracy']:.3f}"
                        elif "rouge_l" in metrics:
                            metrics_str = f" | rouge_l={metrics['rouge_l']:.3f}"
                        elif "f1" in metrics:
                            metrics_str = f" | f1={metrics['f1']:.3f}"

                    summary_str = ""
                    if result.get("summary"):
                        summary = result["summary"]
                        throughput = summary.get("throughput", 0)
                        if throughput > 0:
                            summary_str = f" | {throughput:.1f} tok/s"

                    print(f"{status} {progress} {result['task_str']}{metrics_str}{summary_str}")
                else:
                    print(f"{status} {progress} {result['task_str']} | ERROR")

    # Summary
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)

    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded
    print(f"Total: {len(results)} | Succeeded: {succeeded} | Failed: {failed}")

    if succeeded > 0:
        print("\n" + "-"*80)
        print("RESULTS TABLE")
        print("-"*80)

        # Group by mode
        baselines = [r for r in results if r["success"] and r["task"].mode in ["autoregressive", "eagle_uncapped"]]
        experiments = [r for r in results if r["success"] and r["task"].mode == "eagle_pruned"]

        if baselines:
            print("\nBaselines:")
            print(f"  {'Mode':<20} {'Temp':>6} {'Metric':>10} {'Throughput':>12}")
            print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*12}")
            for r in baselines:
                task = r["task"]
                metric_val = ""
                if r.get("metrics"):
                    m = r["metrics"]
                    if "pass@1" in m:
                        metric_val = f"{m['pass@1']:.3f}"
                    elif "accuracy" in m:
                        metric_val = f"{m['accuracy']:.3f}"
                throughput = r.get("summary", {}).get("throughput", 0)
                throughput_str = f"{throughput:.1f} tok/s" if throughput > 0 else "-"
                print(f"  {task.mode:<20} {task.temperature:>6.1f} {metric_val:>10} {throughput_str:>12}")

        if experiments:
            print("\nExperiments:")
            print(f"  {'Draft':>5} {'Selection':>12} {'Budget':>8} {'Strategy':>12} {'Temp':>6} {'Metric':>10} {'Throughput':>12}")
            print(f"  {'-'*5} {'-'*12} {'-'*8} {'-'*12} {'-'*6} {'-'*10} {'-'*12}")
            for r in experiments:
                task = r["task"]
                budget_str = f"{int(task.budget)}" if task.selection_mode == "cardinality" else f"{task.budget:.2f}"
                metric_val = ""
                if r.get("metrics"):
                    m = r["metrics"]
                    if "pass@1" in m:
                        metric_val = f"{m['pass@1']:.3f}"
                    elif "accuracy" in m:
                        metric_val = f"{m['accuracy']:.3f}"
                throughput = r.get("summary", {}).get("throughput", 0)
                throughput_str = f"{throughput:.1f} tok/s" if throughput > 0 else "-"
                print(f"  {task.draft_tokens:>5} {task.selection_mode:>12} {budget_str:>8} {task.pruning_strategy:>12} {task.temperature:>6.1f} {metric_val:>10} {throughput_str:>12}")

    if failed > 0:
        print("\n" + "-"*80)
        print("FAILED TASKS")
        print("-"*80)
        for r in results:
            if not r["success"]:
                print(f"  - {r['task_str']}")
                if "error" in r:
                    print(f"    Error: {r['error'][:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
