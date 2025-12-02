"""Helpers for discovering MoE-capable layers on wrapped models."""

from __future__ import annotations

from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only needed for type hints
    from eagle.model.ea_model import EaModel


def discover_moe_layers(model: "EaModel") -> List[int]:
    """Return transformer layer indices that expose an `expert_cap` attr."""

    indices: Set[int] = set()
    for module in model.base_model.modules():
        if not hasattr(module, "expert_cap"):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            continue
        try:
            indices.add(int(layer_idx))
        except (TypeError, ValueError):
            continue
    return sorted(indices)
