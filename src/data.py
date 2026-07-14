from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SyntheticTokenDataset(Dataset):
    def __init__(self, total_tokens: int, seq_len: int, vocab_size: int, seed: int = 2026) -> None:
        self.total_tokens = total_tokens
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        self.num_sequences = max(1, total_tokens // (seq_len + 1))

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + index)
        tokens = rng.integers(0, self.vocab_size, size=self.seq_len + 1, dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(tokens[:-1].copy()).long(),
            "labels": torch.from_numpy(tokens[1:].copy()).long(),
        }


class MemmapTokenDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        seq_len: int,
        manifest_file: str = "manifest.json",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        manifest_path = self.data_dir / manifest_file
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.manifest = manifest
        file_name = manifest["train_file"] if split == "train" else manifest["valid_file"]
        token_count = (
            manifest["train_tokens_written"] if split == "train" else manifest["valid_tokens_written"]
        )
        dtype_name = manifest["dtype"]
        if dtype_name != "uint32":
            raise ValueError(f"Unsupported dtype {dtype_name}; expected uint32.")

        self.path = self.data_dir / file_name
        self.tokens = np.memmap(self.path, mode="r", dtype=np.uint32)
        self.token_count = int(token_count)
        self.num_sequences = max(1, (self.token_count - 1) // self.seq_len)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.seq_len
        end = start + self.seq_len + 1
        window = np.asarray(self.tokens[start:end], dtype=np.int64)
        if window.shape[0] != self.seq_len + 1:
            pad = np.zeros(self.seq_len + 1, dtype=np.int64)
            pad[: window.shape[0]] = window
            window = pad
        return {
            "input_ids": torch.from_numpy(window[:-1].copy()).long(),
            "labels": torch.from_numpy(window[1:].copy()).long(),
        }


def build_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    data_cfg = config["data"]
    seq_len = config["train"]["seq_len"]
    if data_cfg["type"] == "synthetic":
        train_tokens = data_cfg.get("train_tokens")
        valid_tokens = data_cfg.get("valid_tokens")
        if train_tokens is None:
            train_tokens = max(
                seq_len * 1024,
                config["train"]["max_steps"]
                * config["train"]["micro_batch_size"]
                * config["train"]["gradient_accumulation_steps"]
                * seq_len
                * 2,
            )
        if valid_tokens is None:
            valid_tokens = max(seq_len * 128, seq_len * 16)
        train_ds = SyntheticTokenDataset(
            total_tokens=train_tokens,
            seq_len=seq_len,
            vocab_size=data_cfg["vocab_size"],
        )
        valid_ds = SyntheticTokenDataset(
            total_tokens=valid_tokens,
            seq_len=seq_len,
            vocab_size=data_cfg["vocab_size"],
            seed=3030,
        )
        return train_ds, valid_ds

    if data_cfg["type"] != "memmap":
        raise ValueError(f"Unsupported data type {data_cfg['type']}")

    data_dir = data_cfg["data_dir"]
    manifest_file = str(data_cfg.get("manifest_file", "manifest.json"))
    train_ds = MemmapTokenDataset(data_dir=data_dir, split="train", seq_len=seq_len, manifest_file=manifest_file)
    valid_ds = MemmapTokenDataset(data_dir=data_dir, split="valid", seq_len=seq_len, manifest_file=manifest_file)
    if "vocab_size" in data_cfg:
        manifest_vocab_size = int(train_ds.manifest["vocab_size"])
        config_vocab_size = int(data_cfg["vocab_size"])
        if config_vocab_size != manifest_vocab_size:
            raise ValueError(
                f"data.vocab_size={config_vocab_size} does not match memmap manifest "
                f"vocab_size={manifest_vocab_size} in {train_ds.data_dir / manifest_file}."
            )
    return train_ds, valid_ds


def build_dataloaders(
    config: dict[str, Any],
    world_size: int,
    rank: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = build_datasets(config)
    data_cfg = config.get("data", {})
    batch_size = config["train"]["micro_batch_size"]

    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_sampler = None
    valid_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=2026)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=2026)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, valid_loader
