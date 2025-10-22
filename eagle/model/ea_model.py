import copy
import json
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .modeling_qwen3_moe_kv import KVQwen3MoeForCausalLM
from .modeling_olmoe_kv import OlmoeForCausalLM as KVOlmoeForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class ExpertTraceCollector:
    """Collects per-layer expert routing data for a single forward pass."""

    def __init__(self) -> None:
        self.records: Dict[int, Dict[str, List]] = {}

    def record(self, layer_idx: int, experts: torch.Tensor, weights: torch.Tensor) -> None:
        experts_cpu = experts.to(torch.int64).cpu()
        weights_cpu = weights.to(torch.float32).cpu()
        self.records[int(layer_idx)] = {
            "experts": experts_cpu.tolist(),
            "weights": weights_cpu.tolist(),
        }

    def to_serializable(self) -> Dict[str, Dict[str, List]]:
        return {str(k): v for k, v in sorted(self.records.items())}


def set_expert_trace_recorder(model: nn.Module, collector: Optional[ExpertTraceCollector]) -> int:
    """Attach or clear expert trace recorders across all supported MoE blocks."""

    updated = 0
    recorder = collector.record if collector is not None else None
    for module in model.modules():
        setter = getattr(module, "set_trace_recorder", None)
        if callable(setter):
            setter(recorder)
            updated += 1
    return updated


def set_token_routing_weights(model: nn.Module, weights: Optional[torch.Tensor]) -> int:
    """Propagate per-token routing weights to all MoE blocks."""

    updated = 0
    for module in model.modules():
        setter = getattr(module, "set_token_routing_weights", None)
        if callable(setter):
            setter(weights)
            updated += 1
    return updated


def compute_subtree_weights(tree_parents: torch.Tensor, tree_position_ids: torch.Tensor) -> torch.Tensor:
    """Return subtree weights for each node (broadcasting over batch if needed)."""

    if tree_parents.dim() == 1:
        return _compute_single_subtree_weights(tree_parents, tree_position_ids)

    weights = []
    for b in range(tree_parents.size(0)):
        weights.append(_compute_single_subtree_weights(tree_parents[b], tree_position_ids[b]))
    return torch.stack(weights, dim=0)


def _compute_single_subtree_weights(tree_parents: torch.Tensor, tree_position_ids: torch.Tensor) -> torch.Tensor:
    num_nodes = tree_parents.numel()
    weights = torch.ones(num_nodes, dtype=torch.float32, device=tree_parents.device)

    depths = tree_position_ids.to(torch.long)
    order = torch.argsort(depths, descending=True)
    for idx in order.tolist():
        parent = int(tree_parents[idx].item())
        if parent >= 0:
            weights[parent] += weights[idx]

    weights /= weights.sum()
    return weights


def summarize_layer_experts(
    layer_record: Dict[str, List],
    node_depths: List[int],
    accepted_nodes: set,
) -> Dict[str, Any]:
    """Build per-layer aggregate statistics over expert routes."""

    experts_list = layer_record.get("experts", []) or []
    weights_list = layer_record.get("weights", []) or []

    def _collapse_batch_axis(values: List) -> List:
        if not isinstance(values, list):
            return []
        if not values:
            return []
        first = values[0]
        if isinstance(first, list):
            if len(values) == 1:
                return first
            flattened: List = []
            for item in values:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return values

    def _flatten_numbers(values) -> List[float]:
        result: List[float] = []
        if isinstance(values, (list, tuple)):
            for item in values:
                result.extend(_flatten_numbers(item))
        elif values is not None:
            try:
                result.append(float(values))
            except (TypeError, ValueError):
                pass
        return result

    per_node_experts = _collapse_batch_axis(experts_list)
    per_node_weights = _collapse_batch_axis(weights_list)

    depth_unique: Dict[int, set] = defaultdict(set)
    accepted_depth_unique: Dict[int, set] = defaultdict(set)
    depth_weight: Dict[int, float] = defaultdict(float)
    accepted_depth_weight: Dict[int, float] = defaultdict(float)

    total_unique: set = set()
    accepted_unique: set = set()
    total_weight = 0.0
    accepted_weight = 0.0

    for node_idx, node_experts in enumerate(per_node_experts):
        if node_experts is None:
            continue

        flat_experts = _flatten_numbers(node_experts)
        if not flat_experts:
            continue

        experts_int = [int(e) for e in flat_experts]
        total_unique.update(experts_int)

        node_depth = node_depths[node_idx] if node_idx < len(node_depths) else None
        if node_depth is not None:
            depth_unique[node_depth].update(experts_int)

        if node_idx in accepted_nodes:
            accepted_unique.update(experts_int)
            if node_depth is not None:
                accepted_depth_unique[node_depth].update(experts_int)

        node_weights = per_node_weights[node_idx] if node_idx < len(per_node_weights) else []
        flat_weights = _flatten_numbers(node_weights)
        if flat_weights:
            weight_sum = float(sum(flat_weights[: len(experts_int)]))
            total_weight += weight_sum
            if node_depth is not None:
                depth_weight[node_depth] += weight_sum
            if node_idx in accepted_nodes:
                accepted_weight += weight_sum
                if node_depth is not None:
                    accepted_depth_weight[node_depth] += weight_sum

    return {
        "total_unique": len(total_unique),
        "accepted_unique": len(accepted_unique),
        "total_unique_ids": sorted(total_unique),
        "accepted_unique_ids": sorted(accepted_unique),
        "depth_unique": {str(k): len(v) for k, v in depth_unique.items() if v},
        "accepted_depth_unique": {str(k): len(v) for k, v in accepted_depth_unique.items() if v},
        "depth_weight": {str(k): depth_weight[k] for k in depth_weight},
        "accepted_depth_weight": {str(k): accepted_depth_weight[k] for k in accepted_depth_weight},
        "total_weight": total_weight,
        "accepted_weight": accepted_weight,
    }


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, trust_remote_code=True)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3MoeForCausalLM':
            base_model = KVQwen3MoeForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'OlmoeForCausalLM':
            base_model = KVOlmoeForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            output_hidden_states: bool = False,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=output_hidden_states,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            collect_expert_traces=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            tree_parents,
            draft_log_probs,
            logits,
            hidden_state,
            sample_token,
        ) = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        accept_lengths = []  # Track acceptance lengths per iteration
        iteration_traces: List[Dict[str, Any]] = []
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            # Snapshot current tree before verification (CPU copies for logging)
            if collect_expert_traces:
                iter_draft_tokens = draft_tokens.detach().to("cpu")
                iter_tree_parents = tree_parents.detach().to("cpu")
                iter_tree_position_ids = tree_position_ids.detach().to("cpu")
                iter_retrieve_indices = retrieve_indices.detach().to("cpu")
                iter_draft_log_probs = draft_log_probs.detach().to("cpu")

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            collector = ExpertTraceCollector() if collect_expert_traces else None
            if collector is not None:
                set_expert_trace_recorder(self.base_model.model, collector)

            routing_weights = None
            subtree_weights_cpu = None
            routing_weights_cpu = None
            try:
                subtree_weights = compute_subtree_weights(tree_parents, tree_position_ids)
                if subtree_weights is not None:
                    if subtree_weights.dim() == 1:
                        subtree_weights = subtree_weights.unsqueeze(0)
                    subtree_weights = subtree_weights.to(draft_tokens.device)
                    routing_weights = subtree_weights.reshape(-1)
                    set_token_routing_weights(self.base_model.model, routing_weights)
                    if collect_expert_traces:
                        subtree_weights_cpu = subtree_weights.detach().to("cpu").tolist()
                        routing_weights_cpu = routing_weights.detach().to("cpu").tolist()
            except Exception:
                set_token_routing_weights(self.base_model.model, None)

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            set_token_routing_weights(self.base_model.model, None)
            if collector is not None:
                set_expert_trace_recorder(self.base_model.model, None)
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            best_candidate_val = (
                int(best_candidate.item()) if isinstance(best_candidate, torch.Tensor) else int(best_candidate)
            )
            accept_length_val = (
                int(accept_length.item()) if isinstance(accept_length, torch.Tensor) else int(accept_length)
            )
            accept_lengths.append(accept_length_val)

            verification_info: Dict[str, Any] = {}
            if collect_expert_traces:
                strategy = "greedy" if logits_processor is None else "posterior"
                logits_cpu = logits.detach().to("cpu")
                logits_norm = float(torch.linalg.vector_norm(logits_cpu).item())
                posterior_slice = logits_cpu[best_candidate_val, : max(1, accept_length_val + 1)]
                verification_info = {
                    "strategy": strategy,
                    "best_candidate": best_candidate_val,
                    "accept_length": accept_length_val,
                    "logits_norm": logits_norm,
                    "posterior_slice": posterior_slice.tolist(),
                }

            if collect_expert_traces:
                accepted_path = iter_retrieve_indices[best_candidate_val, : accept_length_val + 1].tolist()

                iter_tree_depths = iter_tree_position_ids.view(-1)
                tree_depth = int(iter_tree_depths.max().item()) if iter_tree_depths.numel() else 0
                depth_hist = torch.bincount(
                    iter_tree_depths, minlength=tree_depth + 1
                ).tolist()
                total_nodes = int(iter_tree_depths.numel())
                accepted_nodes_set = {n for n in accepted_path if n >= 0}
                accepted_len = len(accepted_nodes_set)

                accepted_experts_summary: Dict[str, Any] = {}
                layer_expert_stats: Dict[str, Any] = {}
                combined_total_unique = 0
                combined_accepted_unique = 0
                if collector is not None:
                    node_depths_list = iter_tree_depths.tolist()
                    for layer_idx, layer_record in collector.records.items():
                        stats = summarize_layer_experts(layer_record, node_depths_list, accepted_nodes_set)
                        layer_key = str(layer_idx)
                        layer_expert_stats[layer_key] = stats
                        accepted_experts_summary[layer_key] = {
                            "unique_count": stats["accepted_unique"],
                            "experts": stats["accepted_unique_ids"],
                        }
                        combined_total_unique += stats["total_unique"]
                        combined_accepted_unique += stats["accepted_unique"]

                pruned_nodes = total_nodes - accepted_len

                node_log_probs: Dict[str, float] = {}
                try:
                    log_probs = torch.log_softmax(logits, dim=-1).detach().to("cpu")
                    if log_probs.dim() == 3:
                        num_cands, seq_len, _ = log_probs.shape
                        for cand_idx in range(num_cands):
                            for pos in range(1, iter_retrieve_indices.size(1)):
                                node_idx = int(iter_retrieve_indices[cand_idx, pos].item())
                                if node_idx < 0:
                                    continue
                                if node_idx >= iter_draft_tokens.size(1):
                                    continue
                                token_id = int(iter_draft_tokens[0, node_idx].item())
                                if token_id < 0 or pos - 1 >= seq_len:
                                    continue
                                logp = float(log_probs[cand_idx, pos - 1, token_id].item())
                                node_log_probs.setdefault(str(node_idx), logp)
                except Exception:
                    node_log_probs = {}

                draft_log_probs_dict: Dict[str, float] = {}
                try:
                    draft_log_probs_list = iter_draft_log_probs.tolist()
                    draft_log_probs_dict = {
                        str(idx): float(val) for idx, val in enumerate(draft_log_probs_list)
                    }
                except Exception:
                    draft_log_probs_dict = {}

                iteration_traces.append(
                    {
                        "iteration": idx,
                        "tokens": iter_draft_tokens[0].tolist(),
                        "parents": iter_tree_parents.tolist(),
                        "depth": iter_tree_position_ids.tolist(),
                        "retrieve_indices": iter_retrieve_indices.tolist(),
                        "accepted_nodes": accepted_path,
                        "tree_total_nodes": total_nodes,
                        "tree_depth": tree_depth,
                        "tree_width_by_depth": depth_hist,
                        "accepted_length": accepted_len,
                        "accepted_unique_experts": accepted_experts_summary,
                        "layer_expert_stats": layer_expert_stats,
                        "total_unique_experts": combined_total_unique,
                        "accepted_unique_experts_count": combined_accepted_unique,
                        "pruned_nodes": pruned_nodes,
                        "subtree_weights": subtree_weights_cpu,
                        "routing_weights": routing_weights_cpu,
                        "verification": verification_info,
                        "node_log_probs": node_log_probs,
                        "draft_node_log_probs": draft_log_probs_dict,
                        "experts": collector.to_serializable() if collector is not None else {},
                    }
                )
            # Adjusting the input sequence, draft model forward
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                tree_parents,
                draft_log_probs,
                new_token,
                hidden_state,
                sample_token,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            if collect_expert_traces:
                set_expert_trace_recorder(self.base_model.model, None)
            return input_ids, new_token, idx, accept_lengths, iteration_traces

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_parents, draft_log_probs, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_parents, draft_log_probs, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
