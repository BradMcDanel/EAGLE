"""Unified evaluation script for EAGLE models across multiple benchmarks.

Supports: mt_bench, gsm8k, humaneval, alpaca, qa, sum

Usage:
python eval_eagle.py \
    --base-model-path allenai/OLMoE-1B-7B-0125-Instruct \
    --ea-model-path wantsleep/OLMoE_1B_7B_Eagle3 \
    --model-id olmoe-1b-eagle3 \
    --conv-template vicuna \
    --bench-name mt_bench \
    --num-questions 10 \
    --use-eagle3
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm

# Ensure repository root is available on sys.path for package imports
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parents[1]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    from fastchat.llm_judge.common import load_questions  # type: ignore
except ImportError:
    def load_questions(path, begin=None, end=None):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        if begin is not None or end is not None:
            begin = begin or 0
            end = end or len(data)
            data = data[begin:end]
        return data

try:
    from fastchat.model import get_conversation_template  # type: ignore
except ImportError:
    def get_conversation_template(template_name):
        raise RuntimeError(
            "fastchat.model is required for conversation templates but is not installed."
        )

try:
    import shortuuid  # type: ignore

    def _make_uuid() -> str:
        return shortuuid.uuid()

except ImportError:
    import uuid

    def _make_uuid() -> str:
        return uuid.uuid4().hex

try:
    from eagle.model.ea_model import EaModel
    from eagle.model.utils import prepare_logits_processor
    from eagle.utils.moe_layers import discover_moe_layers
except ImportError:
    from model.ea_model import EaModel
    from model.utils import prepare_logits_processor

    def discover_moe_layers(_model):  # pragma: no cover - only used in fallback
        raise RuntimeError(
            "discover_moe_layers is unavailable. Run eval_eagle.py from the repository root so the 'eagle' package is importable."
        )


def load_cap_schedule(path: str, target: float) -> List[Optional[int]]:
    with open(path, "r") as f:
        data = json.load(f)
    allocations = data.get("allocations")
    if not allocations:
        raise ValueError(f"No allocations found in schedule file {path}")
    index_map = data.get("allocation_index_by_target") or {}
    target_key = str(int(target))
    idx = index_map.get(target_key)
    if idx is None:
        # Fall back to sequential search on float targets.
        for i, entry in enumerate(allocations):
            if float(entry.get("target_avg_cap")) == float(target):
                idx = i
                break
    if idx is None:
        raise ValueError(f"Target {target} not found in schedule file {path}")
    entry = allocations[idx]
    caps = entry.get("final_caps")
    if not isinstance(caps, list):
        raise ValueError(f"Allocation entry for target {target} missing final_caps in {path}")
    normalized: List[Optional[int]] = []
    for cap in caps:
        if cap is None:
            normalized.append(None)
        else:
            normalized.append(int(cap))
    return normalized


def _validate_cap_target(target: Optional[float]) -> int:
    if target is None:
        raise ValueError("cap_schedule_target must be provided when using cap shapes")
    try:
        int_target = int(target)
    except (TypeError, ValueError) as exc:
        raise ValueError("cap_schedule_target must be an integer for cap shapes") from exc
    if abs(int_target - float(target)) > 1e-6:
        raise ValueError("cap_schedule_target must be an integer value for cap shapes")
    if int_target <= 0:
        raise ValueError("cap_schedule_target must be positive for cap shapes")
    return int_target


def _parse_cap_shape_config(config: Optional[str]) -> Dict[str, Any]:
    if config is None:
        return {}
    config = config.strip()
    if not config:
        return {}
    try:
        data = json.loads(config)
    except json.JSONDecodeError as exc:
        raise ValueError("--cap-shape-config must be valid JSON (e.g., '{\"k_head\":2}')") from exc
    if not isinstance(data, dict):
        raise ValueError("--cap-shape-config must decode to a JSON object")
    return data


def _collect_moe_context(model: EaModel) -> Dict[str, Any]:
    moe_layers = discover_moe_layers(model)
    if not moe_layers:
        raise ValueError("Cap shape scheduling requires a model with MoE layers, but none were found.")
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    schedule_length = 0
    if num_hidden_layers is not None:
        try:
            schedule_length = int(num_hidden_layers)
        except (TypeError, ValueError):
            schedule_length = 0
    schedule_length = max(schedule_length, max(moe_layers) + 1)
    if schedule_length <= 0:
        raise ValueError("Model reports no transformer layers; cannot build cap schedule.")

    top_k: Optional[int] = None
    num_experts: Optional[int] = None
    for module in model.base_model.modules():
        if not hasattr(module, "expert_cap"):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            continue
        try:
            if int(layer_idx) not in moe_layers:
                continue
        except (TypeError, ValueError):
            continue
        if top_k is None and hasattr(module, "top_k"):
            try:
                candidate = int(getattr(module, "top_k"))
                if candidate > 0:
                    top_k = candidate
            except (TypeError, ValueError):
                pass
        if num_experts is None and hasattr(module, "num_experts"):
            try:
                candidate = int(getattr(module, "num_experts"))
                if candidate > 0:
                    num_experts = candidate
            except (TypeError, ValueError):
                pass
        if top_k is not None and num_experts is not None:
            break
    return {
        "moe_layers": moe_layers,
        "schedule_length": schedule_length,
        "top_k": top_k,
        "num_experts": num_experts,
    }


def _resolve_cap_value(
    params: Dict[str, Any],
    label: str,
    target_avg: int,
    ctx: Dict[str, Any],
) -> float:
    value_key = f"cap_{label}"
    scale_key = f"{value_key}_scale"
    min_key = f"{value_key}_min"
    max_key = f"{value_key}_max"

    value: Optional[float] = None
    if value_key in params:
        try:
            value = float(params[value_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{value_key} must be numeric") from exc
    elif scale_key in params:
        try:
            scale = float(params[scale_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{scale_key} must be numeric") from exc
        value = scale * float(target_avg)
    else:
        raise ValueError(
            f"Tapered shape requires either '{value_key}' or '{scale_key}' to be specified"
        )

    if value <= 0:
        raise ValueError(f"{value_key} must be positive after applying scale")

    min_cap = params.get(min_key)
    max_cap = params.get(max_key)
    top_k = ctx.get("top_k")
    num_experts = ctx.get("num_experts")
    if min_cap is None and top_k is not None:
        min_cap = float(top_k)
    if max_cap is None and num_experts is not None:
        max_cap = float(num_experts)

    if min_cap is not None:
        try:
            min_cap = float(min_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{min_key} must be numeric") from exc
        value = max(value, min_cap)
    if max_cap is not None:
        try:
            max_cap = float(max_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{max_key} must be numeric") from exc
        value = min(value, max_cap)

    if value <= 0:
        raise ValueError(f"Resolved {value_key} is non-positive; check scale/min/max settings")

    return value


def _quantize_caps(values: List[float], target_total: int, min_cap: Optional[int], max_cap: Optional[int]) -> List[int]:
    if not values:
        raise ValueError("No values provided for cap quantization")
    floors = [math.floor(v) for v in values]
    remainders = [v - f for v, f in zip(values, floors)]
    diff = int(round(target_total - sum(floors)))
    if diff > 0:
        order = sorted(range(len(floors)), key=lambda i: remainders[i], reverse=True)
        idx = 0
        while diff > 0:
            target_idx = order[idx % len(order)]
            floors[target_idx] += 1
            diff -= 1
            idx += 1
    elif diff < 0:
        order = sorted(range(len(floors)), key=lambda i: remainders[i])
        idx = 0
        while diff < 0:
            target_idx = order[idx % len(order)]
            floors[target_idx] -= 1
            diff += 1
            idx += 1

    for value in floors:
        if min_cap is not None and value < min_cap:
            raise ValueError(
                f"Generated cap {value} is below top_k={min_cap}. Adjust shape parameters or target."
            )
        if max_cap is not None and value > max_cap:
            raise ValueError(
                f"Generated cap {value} exceeds num_experts={max_cap}. Adjust shape parameters or target."
            )
    if sum(floors) != target_total:
        raise ValueError("Unable to match target average after quantization; try different parameters.")
    return floors


def _tapered_caps(target_avg: int, ctx: Dict[str, Any], params: Dict[str, Any]) -> List[int]:
    required = ["k_head", "k_tail"]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Tapered shape requires parameters: {', '.join(missing)}")
    try:
        k_head = int(params["k_head"])
        k_tail = int(params["k_tail"])
    except (TypeError, ValueError) as exc:
        raise ValueError("k_head and k_tail must be integers") from exc
    for label, value in ("k_head", k_head), ("k_tail", k_tail):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")

    ramp_width = params.get("ramp_width", 0)
    try:
        ramp_width = int(ramp_width)
    except (TypeError, ValueError) as exc:
        raise ValueError("ramp_width must be an integer") from exc
    if ramp_width < 0:
        raise ValueError("ramp_width must be non-negative")

    layer_count = len(ctx["moe_layers"])
    head_layers = min(k_head, layer_count)
    tail_layers = min(k_tail, max(layer_count - head_layers, 0))
    if head_layers + tail_layers > layer_count:
        raise ValueError("k_head + k_tail exceeds number of MoE layers")
    middle_layers = layer_count - head_layers - tail_layers
    effective_ramp = 0
    if middle_layers > 0 and ramp_width > 0:
        effective_ramp = min(ramp_width, middle_layers // 2)
    plateau_layers = middle_layers - 2 * effective_ramp
    if plateau_layers < 0:
        plateau_layers = 0

    cap_head = _resolve_cap_value(params, "head", target_avg, ctx)
    cap_tail = _resolve_cap_value(params, "tail", target_avg, ctx)

    target_total = target_avg * layer_count
    head_coeff = head_layers + effective_ramp / 2.0
    tail_coeff = tail_layers + effective_ramp / 2.0
    mid_coeff = plateau_layers + effective_ramp
    numerator = target_total - (cap_head * head_coeff + cap_tail * tail_coeff)
    if mid_coeff <= 0:
        if abs(numerator) > 1e-6:
            raise ValueError(
                "Tapered parameters cannot satisfy the requested average (no middle layers available)."
            )
        cap_mid = None
    else:
        cap_mid = numerator / mid_coeff
        if cap_mid <= 0:
            raise ValueError("Derived cap_mid is non-positive; adjust parameters.")

    def _ramp_values(start: float, end: float, count: int) -> List[float]:
        if count <= 0:
            return []
        step = (end - start) / (count + 1)
        return [start + step * (idx + 1) for idx in range(count)]

    raw_caps: List[float] = []
    raw_caps.extend([float(cap_head)] * head_layers)
    if effective_ramp > 0 and cap_mid is not None:
        raw_caps.extend(_ramp_values(float(cap_head), float(cap_mid), effective_ramp))
    if plateau_layers > 0 and cap_mid is not None:
        raw_caps.extend([float(cap_mid)] * plateau_layers)
    if effective_ramp > 0 and cap_mid is not None:
        raw_caps.extend(_ramp_values(float(cap_mid), float(cap_tail), effective_ramp))
    raw_caps.extend([float(cap_tail)] * tail_layers)

    if len(raw_caps) != layer_count:
        raise ValueError("Internal error: tapered schedule length mismatch")
    if not raw_caps:
        raise ValueError("Tapered schedule produced no caps")

    min_cap = ctx.get("top_k")
    max_cap = ctx.get("num_experts")
    return _quantize_caps(raw_caps, target_total, min_cap, max_cap)


def _linear_caps(target_avg: int, ctx: Dict[str, Any], params: Dict[str, Any]) -> List[int]:
    required = ["cap_min", "cap_max"]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Linear shape requires parameters: {', '.join(missing)}")
    try:
        cap_min = float(params["cap_min"])
        cap_max = float(params["cap_max"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Linear parameters must be numeric") from exc
    if cap_min <= 0 or cap_max <= 0:
        raise ValueError("cap_min and cap_max must be positive")
    if cap_max < cap_min:
        raise ValueError("cap_max must be greater than or equal to cap_min")

    layer_count = len(ctx["moe_layers"])
    if layer_count == 0:
        raise ValueError("No MoE layers found for linear schedule")

    if layer_count == 1:
        raw_caps = [cap_max]
    else:
        center = (layer_count - 1) / 2.0
        span = cap_max - cap_min
        raw_caps = [cap_min + span * (abs(idx - center) / center if center else 0.0) for idx in range(layer_count)]

    target_total = target_avg * layer_count
    raw_sum = sum(raw_caps)
    if raw_sum <= 0:
        raise ValueError("Linear schedule produced non-positive total")
    scale = target_total / raw_sum
    scaled_caps = [value * scale for value in raw_caps]

    min_cap = ctx.get("top_k")
    max_cap = ctx.get("num_experts")
    return _quantize_caps(scaled_caps, target_total, min_cap, max_cap)


def build_cap_shape_schedule(
    model: EaModel,
    shape: str,
    target: Optional[float],
    config: Optional[str],
) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    shape_key = (shape or "").strip().lower()
    if shape_key not in {"flat", "tapered", "linear"}:
        raise ValueError(f"Unsupported cap shape '{shape}'. Choose from flat, tapered, or linear.")

    cap_value = _validate_cap_target(target)
    params = _parse_cap_shape_config(config)
    ctx = _collect_moe_context(model)
    layer_count = len(ctx["moe_layers"])
    if layer_count == 0:
        raise ValueError("Cap shape scheduling requires at least one MoE layer.")

    if shape_key == "flat":
        per_layer_caps = [cap_value] * layer_count
    elif shape_key == "tapered":
        per_layer_caps = _tapered_caps(cap_value, ctx, params)
    elif shape_key == "linear":
        per_layer_caps = _linear_caps(cap_value, ctx, params)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unhandled cap shape '{shape_key}'")

    schedule_length = ctx["schedule_length"]
    caps: List[Optional[int]] = [None] * schedule_length
    for layer_idx, cap in zip(ctx["moe_layers"], per_layer_caps):
        if layer_idx < 0 or layer_idx >= schedule_length:
            raise ValueError(
                f"MoE layer index {layer_idx} is outside the supported range [0, {schedule_length})."
            )
        caps[layer_idx] = int(cap)

    metadata = {
        "shape": shape_key,
        "target_avg_cap": float(cap_value),
        "moe_layers": ctx["moe_layers"],
        "schedule_length": schedule_length,
        "params": params,
        "final_moe_caps": per_layer_caps,
        "achieved_avg_cap": sum(per_layer_caps) / layer_count,
        "cap_bounds": {
            "top_k": ctx.get("top_k"),
            "num_experts": ctx.get("num_experts"),
        },
    }
    return caps, metadata


class AdaptiveCapController:
    """Maintain expert-cap speedups while respecting an acceptance target."""

    def __init__(
        self,
        *,
        min_cap: int,
        max_cap: int,
        target_accept: float,
        tolerance: float,
        step: int,
        window: int,
    ) -> None:
        if max_cap <= min_cap:
            raise ValueError("max_cap must be greater than min_cap for adaptive control.")

        self.min_cap = int(min_cap)
        self.max_cap = int(max_cap)
        self.step = max(1, int(step))
        self.target_accept = float(target_accept)
        self.tolerance = max(0.0, float(tolerance))
        self.window = max(1, int(window))
        self.cooldown_steps = self.window

        self.current_cap = self.min_cap
        self.accept_window: Deque[float] = deque(maxlen=self.window)
        self.cooldown = 0

        self.total_iterations = 0
        self.cap_hist: Dict[int, int] = {}
        self.adjustments = 0

        # Sequence-scoped stats (reset every on_sequence_start)
        self.seq_iterations = 0
        self.seq_cap_sum = 0.0
        self.seq_accept_sum = 0.0
        self.seq_cap_hist: Dict[int, int] = {}
        self.seq_adjustments = 0
        self.wave_log: List[Dict[str, Any]] = []
        self.last_summary: Dict[str, Any] = {}

    def on_sequence_start(self) -> None:
        self.current_cap = self.min_cap
        self.accept_window.clear()
        self.cooldown = 0
        self.seq_iterations = 0
        self.seq_cap_sum = 0.0
        self.seq_accept_sum = 0.0
        self.seq_cap_hist = {}
        self.seq_adjustments = 0
        self.wave_log = []

    def _record_cap(self, cap: int) -> None:
        self.cap_hist[cap] = self.cap_hist.get(cap, 0) + 1
        self.seq_cap_hist[cap] = self.seq_cap_hist.get(cap, 0) + 1

    def _build_summary(self) -> Dict[str, Any]:
        seq_avg_cap = (
            self.seq_cap_sum / self.seq_iterations if self.seq_iterations else float(self.current_cap)
        )
        seq_avg_accept = (
            self.seq_accept_sum / self.seq_iterations if self.seq_iterations else 0.0
        )
        return {
            "min_cap": self.min_cap,
            "max_cap": self.max_cap,
            "step": self.step,
            "target_accept": self.target_accept,
            "tolerance": self.tolerance,
            "window": self.window,
            "current_cap": self.current_cap,
            "sequence_iterations": self.seq_iterations,
            "sequence_avg_cap": seq_avg_cap,
            "sequence_avg_accept": seq_avg_accept,
            "sequence_cap_hist": dict(self.seq_cap_hist),
            "sequence_adjustments": self.seq_adjustments,
            "total_iterations": self.total_iterations,
            "total_cap_hist": dict(self.cap_hist),
            "total_adjustments": self.adjustments,
            "recent_accept_window": list(self.accept_window),
        }

    def on_iteration_end(
        self,
        iteration_idx: int,
        metrics: Dict[str, Any],
    ) -> Optional[int]:
        cap_used = int(metrics.get("cap") or self.current_cap)
        accept = float(metrics.get("accept_length") or 0.0)

        self.total_iterations += 1
        self.seq_iterations += 1
        self.seq_cap_sum += cap_used
        self.seq_accept_sum += accept
        self._record_cap(cap_used)

        self.accept_window.append(accept)
        avg_window = sum(self.accept_window) / len(self.accept_window)

        lower_bound = self.target_accept - self.tolerance
        upper_bound = self.target_accept + self.tolerance
        urgent_lower_bound = lower_bound - self.tolerance
        urgent_upper_bound = upper_bound + self.tolerance

        requested_cap: Optional[int] = None
        action = "hold"

        # Urgent adjustments react immediately to extreme deviations.
        if accept < urgent_lower_bound and self.current_cap < self.max_cap:
            requested_cap = min(self.max_cap, self.current_cap + self.step)
            action = "raise"
        elif accept > urgent_upper_bound and self.current_cap > self.min_cap:
            requested_cap = max(self.min_cap, self.current_cap - self.step)
            action = "lower"
        else:
            if self.cooldown > 0:
                self.cooldown -= 1
            if self.cooldown == 0 and len(self.accept_window) == self.window:
                if avg_window < lower_bound and self.current_cap < self.max_cap:
                    requested_cap = min(self.max_cap, self.current_cap + self.step)
                    action = "raise"
                elif avg_window > upper_bound and self.current_cap > self.min_cap:
                    requested_cap = max(self.min_cap, self.current_cap - self.step)
                    action = "lower"

        if requested_cap is not None and requested_cap == self.current_cap:
            requested_cap = None
            action = "hold"

        if requested_cap is not None:
            self.current_cap = requested_cap
            self.accept_window.clear()
            self.cooldown = self.cooldown_steps
            self.adjustments += 1
            self.seq_adjustments += 1

        log_entry = {
            "iteration": iteration_idx,
            "cap": cap_used,
            "next_cap": self.current_cap,
            "accept_length": accept,
            "window_avg_accept": avg_window,
            "action": action,
            "cooldown": self.cooldown,
            "iteration_time": metrics.get("iteration_time"),
            "margin": metrics.get("margin"),
        }
        self.wave_log.append(log_entry)

        return self.current_cap if requested_cap is not None else None

    def on_sequence_end(self) -> None:
        # Nothing to reset; capture summary for posterity.
        self.last_summary = self._build_summary()

    def summary(self) -> Dict[str, Any]:
        self.last_summary = self._build_summary()
        return dict(self.last_summary)

    def pop_wave_log(self) -> List[Dict[str, Any]]:
        log = list(self.wave_log)
        self.wave_log.clear()
        return log


@torch.inference_mode()
def run_evaluation(
    base_model_path,
    ea_model_path,
    model_id,
    conv_template,
    question_file,
    answer_file,
    num_questions,
    max_new_tokens,
    temperature,
    use_eagle,
    args
):
    """Run MT-Bench evaluation with optional EAGLE acceleration."""

    # Load model
    print(f"Loading model from {base_model_path}")
    print(f"EAGLE adapter: {ea_model_path if use_eagle else 'None (baseline)'}")

    # Always load via EaModel wrapper (provides optimized KV-cache implementation)
    # Baseline mode uses naivegenerate(), EAGLE mode uses eagenerate()

    # For baseline, we need a dummy EAGLE checkpoint path to initialize the wrapper
    # We can use any EAGLE checkpoint since we only use the base model
    if not use_eagle:
        # Check if OLMoE - if so, use a default EAGLE checkpoint for initialization
        from transformers import AutoConfig
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'OlmoeForCausalLM' and not ea_model_path:
            # Use default EAGLE3 checkpoint for OLMoE (only to initialize wrapper, not used in baseline)
            ea_model_path = 'wantsleep/OLMoE_1B_7B_Eagle3'
            args.use_eagle3 = True  # Match the checkpoint architecture
            print(f"Note: Using {ea_model_path} for wrapper initialization (only base model will be used)")

    if not ea_model_path:
        raise ValueError("--ea-model-path is required. For baseline, the same checkpoint path is needed to initialize the wrapper.")

    model = EaModel.from_pretrained(
        use_eagle3=args.use_eagle3,
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token if use_eagle else 60,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    cap_shape_metadata: Optional[Dict[str, Any]] = None
    adaptive_enabled = False
    adaptive_controller: Optional[AdaptiveCapController] = None
    if args.cap_schedule_file:
        caps = load_cap_schedule(args.cap_schedule_file, args.cap_schedule_target)
        model.set_expert_cap_schedule(caps)
        print(
            f"Applying cap schedule {args.cap_schedule_file} "
            f"(target={args.cap_schedule_target}, layers={len(caps)})"
        )
    elif args.cap_shape:
        caps, metadata = build_cap_shape_schedule(
            model,
            args.cap_shape,
            args.cap_schedule_target,
            args.cap_shape_config,
        )
        model.set_expert_cap_schedule(caps)
        num_layers = len(metadata.get("moe_layers", []))
        print(
            f"Applying {args.cap_shape} cap schedule (target={metadata['target_avg_cap']}, "
            f"moe_layers={num_layers}, params={metadata.get('params', {})})"
        )
        cap_shape_metadata = metadata
        setattr(args, "cap_shape_metadata", metadata)
    else:
        if use_eagle and args.adaptive_cap_min is not None and args.adaptive_cap_max is not None:
            adaptive_enabled = True
            model.set_expert_cap(args.adaptive_cap_min)
            print(
                "Adaptive expert cap enabled: "
                f"min={args.adaptive_cap_min}, max={args.adaptive_cap_max}, "
                f"target_accept={args.adaptive_target_accept}, tolerance={args.adaptive_tolerance}, "
                f"step={args.adaptive_step}, window={args.adaptive_window}"
            )
            adaptive_controller = AdaptiveCapController(
                min_cap=args.adaptive_cap_min,
                max_cap=args.adaptive_cap_max,
                step=args.adaptive_step,
                target_accept=args.adaptive_target_accept,
                tolerance=args.adaptive_tolerance,
                window=args.adaptive_window,
            )
        else:
            model.set_expert_cap(args.expert_cap)
            if args.expert_cap is not None:
                print(f"Applying expert cap: {args.expert_cap}")
    oracle_enabled = False
    if args.oracle_trace_file:
        oracle_enabled = model.configure_oracle(
            trace_file=args.oracle_trace_file,
            choice_index=args.oracle_choice_index,
            strict=args.oracle_strict,
        )
        trace_count = len(model.oracle.trace_map) if model.oracle is not None else 0
        if oracle_enabled:
            print(f"Oracle trace replay enabled ({trace_count} turn traces loaded from {args.oracle_trace_file}).")
        else:
            print(
                f"Warning: Oracle trace file '{args.oracle_trace_file}' loaded but no matching traces were found."
            )

    tokenizer = model.get_tokenizer()

    # Detect if this is a Llama-3, OLMoE, or Qwen3 model (all use tokenizer chat templates)
    is_llama3 = "llama-3" in base_model_path.lower() or "llama3" in base_model_path.lower()
    is_olmoe = "olmoe" in base_model_path.lower()
    is_qwen3 = "qwen3" in base_model_path.lower()
    use_chat_template = is_llama3 or is_olmoe or is_qwen3

    if is_llama3:
        print("Detected Llama-3 model - will use tokenizer chat template and special stop token handling")
    elif is_olmoe:
        print("Detected OLMoE model - will use tokenizer chat template")
    elif is_qwen3:
        print("Detected Qwen3 model - will use tokenizer chat template")

    # Load questions
    questions = load_questions(question_file, args.question_begin, args.question_end)
    if num_questions:
        questions = questions[:num_questions]

    print(f"Evaluating {len(questions)} questions with temperature={temperature}")
    print(f"Conversation template: {conv_template}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Warmup tokens: {args.warmup_tokens}")

    training_rows: List[Dict[str, Any]] = []
    training_parquet_path: Optional[str] = None
    if args.collect_expert_traces and args.trace_schema == "training":
        base_name, _ = os.path.splitext(answer_file)
        training_parquet_path = base_name + ".training.parquet"
        try:
            os.remove(training_parquet_path)
        except FileNotFoundError:
            pass

    def append_training_rows(
        traces: Optional[List[Dict[str, Any]]],
        meta: Dict[str, Any],
    ) -> None:
        if traces is None:
            return
        for trace in traces:
            if not isinstance(trace, dict) or trace.get("schema") != "training":
                continue
            iteration_idx = trace.get("iteration")
            node_features = trace.get("node_features") or []
            for node_entry in node_features:
                row = dict(meta)
                row["iteration"] = iteration_idx
                row["wave_index"] = iteration_idx
                row["trace_schema"] = "training"
                row.update(node_entry)
                training_rows.append(row)

    # Warmup
    print("Warming up (2 cycles)...")
    question = questions[0]
    for warmup_idx in range(2):
        print(f"  Warmup cycle {warmup_idx + 1}/2...")
        if use_chat_template:
            # For models with native chat templates (Llama-3, OLMoE), use tokenizer chat template
            messages = []
        else:
            conv = get_conversation_template(conv_template)

        for turn_idx, turn in enumerate(question["turns"]):
            print(f"    Turn {turn_idx + 1}/{len(question['turns'])}...", end=" ", flush=True)

            wave_metrics: List[Dict[str, Any]] = []
            if use_chat_template:
                messages.append({"role": "user", "content": turn})
                # For Qwen3, disable thinking mode for faster, more direct responses
                enable_thinking = False if is_qwen3 else None
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
            else:
                conv.append_message(conv.roles[0], turn)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

            start = time.time()
            if use_eagle:
                output_ids, _, _, _, _ = model.eagenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=args.warmup_tokens,
                    log=True,
                    is_llama3=is_llama3,
                    collect_expert_traces=args.collect_expert_traces and args.trace_schema == "analysis",
                    trace_schema="analysis",
                )
            else:
                # Baseline: use naivegenerate (optimized AR decoding)
                output_ids, _, _ = model.naivegenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=args.warmup_tokens,
                    log=True,
                    is_llama3=is_llama3
                )

            output = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            elapsed = time.time() - start
            print(f"done ({elapsed:.2f}s)")

            if use_chat_template:
                messages.append({"role": "assistant", "content": output})
            else:
                conv.messages[-1][-1] = output

    print("Warmup complete. Starting evaluation...")

    # Track statistics
    all_stats = {
        'total_tokens': [],
        'total_iterations': [],
        'wall_time': [],
        'tokens_per_second': [],
    }

    if use_eagle:
        all_stats['accept_lengths'] = []
        all_stats['avg_accept_length'] = []
        all_stats['tokens_per_iter'] = []
        all_stats['speedup'] = []

    # Evaluate questions
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    for record_index, question in enumerate(tqdm(questions, desc="Evaluating")):
        torch.manual_seed(0)
        if use_chat_template:
            messages = []
        else:
            conv = get_conversation_template(conv_template)

        turns = []
        turns_stats = []

        for turn_idx, turn_text in enumerate(question["turns"]):
            if use_chat_template:
                messages.append({"role": "user", "content": turn_text})
                # For Qwen3, disable thinking mode for faster, more direct responses
                enable_thinking = False if is_qwen3 else None
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
            else:
                conv.append_message(conv.roles[0], turn_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

            torch.cuda.synchronize()
            start_time = time.time()

            oracle_turn_active = False
            if use_eagle and model.has_oracle():
                oracle_turn_active = model.start_oracle_turn(question.get("question_id"), turn_idx)

            if use_eagle:
                controller_for_turn = adaptive_controller if adaptive_enabled else None
                if controller_for_turn is not None:
                    controller_for_turn.on_sequence_start()
                output_ids, new_tokens, iterations, accept_lengths, iteration_traces = model.eagenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    log=True,
                    is_llama3=is_llama3,
                    collect_expert_traces=args.collect_expert_traces,
                    trace_schema=args.trace_schema,
                    adaptive_controller=controller_for_turn,
                )
            else:
                # Baseline: use naivegenerate (optimized AR decoding)
                output_ids, new_tokens, iterations = model.naivegenerate(
                    torch.as_tensor(input_ids).to(model.base_model.device),
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    log=True,
                    is_llama3=is_llama3
                )
                accept_lengths = None

            oracle_stats = None
            if use_eagle and model.has_oracle():
                oracle_stats = model.finish_oracle_turn()

            # Decode output
            output_ids = output_ids[0][len(input_ids[0]):]

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            # Handle stop tokens (for models not using chat templates)
            if not use_chat_template and conv.stop_token_ids:
                stop_indices = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                if stop_indices:
                    output_ids = output_ids[:stop_indices[0]]

            output = tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)

            # Clean up output (for models not using chat templates)
            if not use_chat_template and conv.stop_str and conv.stop_str in output:
                output = output[:output.find(conv.stop_str)]

            output = output.strip()
            turns.append(output)

            # Track statistics
            # Ensure all values are Python types, not tensors
            new_tokens_val = new_tokens.item() if isinstance(new_tokens, torch.Tensor) else int(new_tokens)
            iterations_val = iterations.item() if isinstance(iterations, torch.Tensor) else int(iterations)

            throughput = new_tokens_val / elapsed if elapsed > 0 else 0
            turn_stats = {
                'tokens': new_tokens_val,
                'iterations': iterations_val,
                'time': elapsed,
                'throughput': throughput,
            }

            if use_eagle and oracle_stats and oracle_stats.get("expected_iterations", 0) > 0:
                oracle_payload = dict(oracle_stats)
                oracle_payload["turn_active"] = bool(oracle_turn_active)
                turn_stats['oracle'] = oracle_payload

            if use_eagle and accept_lengths is not None:
                # Convert list of tensors to CPU if needed
                if isinstance(accept_lengths, list):
                    accept_lengths = [x.item() if isinstance(x, torch.Tensor) else x for x in accept_lengths]
                elif isinstance(accept_lengths, torch.Tensor):
                    accept_lengths = accept_lengths.cpu().tolist()

                avg_accept = np.mean(accept_lengths)
                # Tokens per iteration = accepted tokens + 1 (the base token)
                tokens_per_iter = avg_accept + 1
                # Speedup vs baseline
                speedup = new_tokens_val / iterations_val if iterations_val > 0 else 0

                turn_stats['avg_accept_length'] = float(avg_accept)
                turn_stats['tokens_per_iter'] = float(tokens_per_iter)
                turn_stats['speedup'] = float(speedup)
                turn_stats['accept_lengths'] = [int(x) for x in accept_lengths]
                if args.collect_expert_traces:
                    turn_stats['trace_schema'] = args.trace_schema
                    if args.trace_schema == "training" and training_parquet_path is not None:
                        meta = {
                            "dataset": args.bench_name,
                            "trace_file": training_parquet_path,
                            "record_index": record_index,
                            "choice_index": turn_idx,
                            "stats_index": turn_idx,
                            "question_id": question.get("question_id"),
                            "model_id": model_id,
                        }
                        append_training_rows(iteration_traces, meta)
                        turn_stats['trace_pointer'] = training_parquet_path
                        turn_stats['expert_traces'] = []
                    else:
                        turn_stats['expert_traces'] = iteration_traces
                if adaptive_enabled and controller_for_turn is not None:
                    summary = controller_for_turn.summary()
                    turn_stats['adaptive_cap'] = summary
                    wave_metrics = controller_for_turn.pop_wave_log()
                    if wave_metrics:
                        turn_stats['wave_metrics'] = wave_metrics

            turns_stats.append(turn_stats)

            # Update conversation
            if use_chat_template:
                messages.append({"role": "assistant", "content": output})
            else:
                conv.messages[-1][-1] = output

        # Save answer
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": _make_uuid(),
            "model_id": model_id,
            "choices": [{
                "index": 0,
                "turns": turns,
                "stats": turns_stats
            }],
            "tstamp": time.time(),
        }

        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans_json) + "\n")

        for turn_stats in turns_stats:
            all_stats['total_tokens'].append(turn_stats['tokens'])
            all_stats['total_iterations'].append(turn_stats['iterations'])
            all_stats['wall_time'].append(turn_stats['time'])
            all_stats['tokens_per_second'].append(turn_stats['throughput'])

            if use_eagle:
                all_stats['avg_accept_length'].append(turn_stats['avg_accept_length'])
                all_stats['tokens_per_iter'].append(turn_stats['tokens_per_iter'])
                all_stats['speedup'].append(turn_stats['speedup'])

    if training_rows and training_parquet_path is not None:
        df = pd.DataFrame(training_rows)
        df.to_parquet(training_parquet_path, index=False)
        print(f"Wrote training traces to {training_parquet_path} ({len(df)} rows)")

    # Print summary statistics
    print("\n" + "="*70)
    print(f"EVALUATION SUMMARY - {'EAGLE' if use_eagle else 'BASELINE'}")
    print("="*70)
    print(f"Questions evaluated: {len(questions)}")
    print(f"Total turns: {len(all_stats['total_tokens'])}")
    print(f"\nAggregate Statistics:")
    print(f"  Total tokens generated: {sum(all_stats['total_tokens'])}")
    print(f"  Total iterations: {sum(all_stats['total_iterations'])}")
    print(f"  Total time: {sum(all_stats['wall_time']):.2f}s")
    print(f"  Mean throughput: {np.mean(all_stats['tokens_per_second']):.2f} tokens/s")
    print(f"  Median throughput: {np.median(all_stats['tokens_per_second']):.2f} tokens/s")

    if use_eagle:
        print(f"\nEAGLE Statistics:")
        print(f"  Mean acceptance ratio: {np.mean(all_stats['tokens_per_iter']):.2f}")
        print(f"  Total tokens target: {args.total_token}")

    print("="*70)
    print(f"\nResults saved to: {answer_file}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation script for EAGLE models across multiple benchmarks")

    # Model arguments
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path or HF repo ID for base model")
    parser.add_argument("--ea-model-path", type=str, default=None,
                        help="Path or HF repo ID for EAGLE adapter (optional for baseline)")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Model identifier for output file")
    parser.add_argument("--conv-template", type=str, default="vicuna",
                        help="Conversation template (vicuna, llama-2, llama-3, etc.)")

    # Evaluation arguments
    parser.add_argument("--bench-name", type=str, default="mt_bench",
                        help="Benchmark name")
    parser.add_argument("--question-file", type=str, default=None,
                        help="Path to question file (default: auto-detect from bench-name)")
    parser.add_argument("--answer-file", type=str, default=None,
                        help="Path to answer file (default: auto-generate)")
    parser.add_argument("--question-begin", type=int, default=None,
                        help="A debug option. The begin index of questions.")
    parser.add_argument("--question-end", type=int, default=None,
                        help="A debug option. The end index of questions.")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Number of questions to evaluate (default: all, ignored if question-begin/end used)")

    # Generation arguments
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum number of new tokens")
    parser.add_argument("--warmup-tokens", type=int, default=64,
                        help="Number of tokens to generate during warmup (default: 64)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")

    # EAGLE arguments
    parser.add_argument("--use-eagle", action="store_true",
                        help="Use EAGLE acceleration (requires --ea-model-path)")
    parser.add_argument("--use-eagle3", action="store_true",
                        help="Use EAGLE-3 mode")
    parser.add_argument("--total-token", type=int, default=63,
                        help="Number of draft tokens for EAGLE")
    parser.add_argument("--collect-expert-traces", action="store_true",
                        help="Capture per-iteration draft tree and routing traces (adds overhead)")
    parser.add_argument("--trace-schema", choices=["analysis", "training"], default="analysis",
                        help="Trace export format when collecting expert traces (analysis = full tree, training = lean node features)")
    parser.add_argument("--expert-cap", type=int, default=None,
                        help="Limit the number of experts evaluated per layer (applies to base/verify model)")
    parser.add_argument("--cap-schedule-file", type=str, default=None,
                        help="Path to JSON schedule produced by allocate_layer_caps.py")
    parser.add_argument("--cap-schedule-target", type=float, default=None,
                        help="Target average cap to load from the schedule file")
    parser.add_argument("--cap-shape", type=str, choices=["flat", "tapered", "linear"], default=None,
                        help="Generate per-layer caps using a preset shape")
    parser.add_argument("--cap-shape-config", type=str, default=None,
                        help="JSON object with shape-specific parameters (e.g., '{\"k_head\":2,...}')")
    parser.add_argument("--adaptive-cap-min", type=int, default=None,
                        help="Enable adaptive expert capping with this minimum cap")
    parser.add_argument("--adaptive-cap-max", type=int, default=None,
                        help="Maximum cap the controller may reach")
    parser.add_argument("--adaptive-step", type=int, default=2,
                        help="Cap increment/decrement applied when the controller reacts")
    parser.add_argument("--adaptive-target-accept", type=float, default=None,
                        help="Acceptance length the controller tries to maintain")
    parser.add_argument("--adaptive-tolerance", type=float, default=0.2,
                        help="Deadband around the target acceptance to avoid oscillation")
    parser.add_argument("--adaptive-window", type=int, default=4,
                        help="Number of iterations to average acceptance over before adjusting")
    parser.add_argument("--oracle-trace-file", type=str, default=None,
                        help="Path to a JSONL answer log containing expert_traces for oracle replay pruning")
    parser.add_argument("--oracle-choice-index", type=int, default=0,
                        help="Choice index to read from when loading oracle traces")
    parser.add_argument("--oracle-strict", action="store_true",
                        help="Fail if oracle trace for a turn is missing or exhausted during replay")

    args = parser.parse_args()

    if args.trace_schema == "training" and not args.collect_expert_traces:
        parser.error("--trace-schema=training requires --collect-expert-traces")
    if args.cap_schedule_file and args.cap_schedule_target is None:
        parser.error("--cap-schedule-target is required when --cap-schedule-file is provided")
    if args.cap_schedule_file and args.expert_cap is not None:
        parser.error("Use either --expert-cap or --cap-schedule-file, not both")
    if args.cap_shape and args.cap_schedule_file:
        parser.error("Use either --cap-shape or --cap-schedule-file, not both")
    if args.cap_shape and args.expert_cap is not None:
        parser.error("Use either --cap-shape or --expert-cap, not both")
    if args.cap_shape and args.cap_schedule_target is None:
        parser.error("--cap-schedule-target is required when --cap-shape is provided")
    if args.cap_shape_config and not args.cap_shape:
        parser.error("--cap-shape-config requires --cap-shape")

    adaptive_min = args.adaptive_cap_min
    adaptive_max = args.adaptive_cap_max
    if (adaptive_min is None) ^ (adaptive_max is None):
        parser.error("--adaptive-cap-min and --adaptive-cap-max must be provided together")
    if adaptive_min is not None:
        if args.cap_schedule_file or args.cap_shape:
            parser.error("Adaptive cap cannot be combined with --cap-schedule-file or --cap-shape")
        if adaptive_max <= adaptive_min:
            parser.error("--adaptive-cap-max must be greater than --adaptive-cap-min")
        if not args.use_eagle:
            parser.error("Adaptive cap requires --use-eagle")
        if args.adaptive_step <= 0:
            parser.error("--adaptive-step must be positive")
        if args.adaptive_target_accept is None or args.adaptive_target_accept <= 0:
            parser.error("--adaptive-target-accept must be provided and positive")
        if args.adaptive_tolerance < 0:
            parser.error("--adaptive-tolerance must be non-negative")
        if args.adaptive_window <= 0:
            parser.error("--adaptive-window must be positive")

    # Validate arguments
    if args.use_eagle and not args.ea_model_path:
        parser.error("--use-eagle requires --ea-model-path")

    # Setup paths
    if args.question_file is None:
        args.question_file = os.path.join(parent_dir, f"data/{args.bench_name}/question.jsonl")

    if args.answer_file is None:
        mode_suffix = "eagle" if args.use_eagle else "baseline"
        args.answer_file = f"results/{args.bench_name}/{args.model_id}-{mode_suffix}-t{args.temperature}.jsonl"

    # Print configuration
    print("="*70)
    print("CONFIGURATION:")
    print("="*70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("="*70)

    # Run evaluation
    run_evaluation(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        model_id=args.model_id,
        conv_template=args.conv_template,
        question_file=args.question_file,
        answer_file=args.answer_file,
        num_questions=args.num_questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_eagle=args.use_eagle,
        args=args
    )
