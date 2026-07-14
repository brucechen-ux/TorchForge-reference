from __future__ import annotations

import importlib
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from src.modeling_v3 import DeepSeekV3LikeLM
from src.utils import load_yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def apply_config_overrides(
    config: dict[str, Any],
    *,
    data_dir: str | None = None,
    seq_len: int | None = None,
) -> dict[str, Any]:
    config = deepcopy(config)
    if data_dir:
        config.setdefault("data", {})["data_dir"] = data_dir
    if seq_len is not None:
        config["train"]["seq_len"] = int(seq_len)
        config["model"]["seq_len"] = int(seq_len)
        if "v4_attention" in config:
            attn_cfg = config.setdefault("v4_attention", {})
            attn_cfg["max_position_embeddings"] = max(int(attn_cfg.get("max_position_embeddings", 0)), int(seq_len))
        if "mla" in config:
            mla_cfg = config.setdefault("mla", {})
            mla_cfg["max_position_embeddings"] = max(int(mla_cfg.get("max_position_embeddings", 0)), int(seq_len))
    return config


def load_config(
    config_path: str | Path | None,
    *,
    checkpoint_config: dict[str, Any] | None = None,
    data_dir: str | None = None,
    seq_len: int | None = None,
) -> dict[str, Any]:
    if config_path is not None:
        config = load_yaml(config_path)
    elif checkpoint_config is not None:
        config = checkpoint_config
    else:
        raise ValueError("--config is required when checkpoint metadata does not include a config")
    return apply_config_overrides(config, data_dir=data_dir, seq_len=seq_len)


def _read_latest_tag(checkpoint_root: Path) -> str:
    latest_path = checkpoint_root / "latest"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Unable to find DeepSpeed latest tag file at {latest_path}")
    tag = latest_path.read_text(encoding="utf-8").strip()
    if not tag:
        raise ValueError(f"DeepSpeed latest tag file is empty: {latest_path}")
    return tag


def _is_zero_tag_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*_optim_states.pt")) and any(path.glob("*_model_states.pt"))


def _normalize_zero_checkpoint_path(checkpoint_path: Path) -> tuple[Path, str | None]:
    if checkpoint_path.is_file() and checkpoint_path.name == "latest":
        return checkpoint_path.parent, None
    if _is_zero_tag_dir(checkpoint_path):
        return checkpoint_path.parent, checkpoint_path.name
    return checkpoint_path, None


def _load_zero_checkpoint_metadata(checkpoint_root: Path, tag: str) -> dict[str, Any]:
    state_path = checkpoint_root / tag / "mp_rank_00_model_states.pt"
    if not state_path.is_file():
        return {}
    state = torch.load(state_path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(state, dict):
        metadata["step"] = int(state.get("step", state.get("global_steps", 0)))
        if isinstance(state.get("config"), dict):
            metadata["config"] = state["config"]
    return metadata


def detect_checkpoint_format(checkpoint: str | Path) -> str:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_file():
        if checkpoint_path.name == "latest":
            return "deepspeed_zero"
        return "plain_pt"
    if checkpoint_path.is_dir():
        if _is_zero_tag_dir(checkpoint_path):
            return "deepspeed_zero"
        if (checkpoint_path / "latest").is_file():
            return "deepspeed_zero"
    raise FileNotFoundError(f"Unsupported checkpoint input: {checkpoint_path}")


def _candidate_deepspeed_paths() -> list[Path]:
    root = repo_root()
    candidates: list[Path] = []
    raw_env = os.environ.get("DEEPSPEED_PYTHONPATH", "")
    if raw_env:
        for item in raw_env.split(os.pathsep):
            if item:
                candidates.append(Path(item).expanduser())
    candidates.extend(
        [
            root / ".deps",
            root.parent / "deepseek-v3-tiny-code-0608" / ".deps",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _import_zero_to_fp32():
    try:
        return importlib.import_module("deepspeed.utils.zero_to_fp32")
    except ImportError:
        pass

    for candidate in _candidate_deepspeed_paths():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            return importlib.import_module("deepspeed.utils.zero_to_fp32")
        except ImportError:
            continue

    search_roots = ", ".join(str(path) for path in _candidate_deepspeed_paths()) or "<none>"
    raise RuntimeError(
        "DeepSpeed checkpoint loading requires importable deepspeed utilities. "
        f"Tried DEEPSPEED_PYTHONPATH and local candidates: {search_roots}"
    )


def load_state_dict_from_checkpoint(checkpoint: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(checkpoint).expanduser()
    checkpoint_format = detect_checkpoint_format(checkpoint_path)

    if checkpoint_format == "plain_pt":
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"Checkpoint file does not contain a 'model' state dict: {checkpoint_path}")
        metadata = {
            "checkpoint_format": "plain_pt",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "step": int(payload.get("step", 0)),
        }
        if isinstance(payload.get("config"), dict):
            metadata["config"] = payload["config"]
        return payload["model"], metadata

    checkpoint_root, tag = _normalize_zero_checkpoint_path(checkpoint_path)
    if tag is None:
        tag = _read_latest_tag(checkpoint_root)
    metadata = _load_zero_checkpoint_metadata(checkpoint_root, tag)
    zero_to_fp32 = _import_zero_to_fp32()
    state_dict = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_root), tag=tag)
    metadata.update(
        {
        "checkpoint_format": "deepspeed_zero",
        "checkpoint_root": str(checkpoint_root.resolve()),
        "checkpoint_tag": tag,
        "checkpoint_path": str((checkpoint_root / tag).resolve()),
        }
    )
    return state_dict, metadata


def public_checkpoint_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "config"}


def build_model_from_checkpoint(
    config_path: str | Path | None,
    checkpoint: str | Path,
    device: torch.device,
    *,
    data_dir: str | None = None,
    seq_len: int | None = None,
) -> tuple[DeepSeekV3LikeLM, dict[str, Any], dict[str, Any]]:
    state_dict, metadata = load_state_dict_from_checkpoint(checkpoint)
    config = load_config(
        config_path,
        checkpoint_config=metadata.get("config"),
        data_dir=data_dir,
        seq_len=seq_len,
    )
    model = DeepSeekV3LikeLM(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, metadata
