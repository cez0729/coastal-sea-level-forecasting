from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SEEDS = (42, 123, 2024, 2025, 3407)


@dataclass(frozen=True)
class TierAContract:
    n_features: int = 34
    input_hours: int = 24
    forecast_hours: int = 24
    test_windows: int = 3899
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    train_stride: int = 8
    hidden_dim: int = 64
    gwn_blocks: int = 6
    diffusion_steps: int = 2
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 120
    patience: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    aux_weight: float = 0.08
    last_step_weight: float = 0.20
    physics_lambda: float = 2e-4
    physics_warmup_epochs: int = 8
    physics_ramp_epochs: int = 14
    multistate_wave_target: str = "wave_setup_proxy"
    fixed_graph_type: str = "distance"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def load_official_modules(tiera_root_text: str):
    root = Path(tiera_root_text).resolve()
    scripts = root
    required = [
        root / "PROTOCOL.json",
        REPO_ROOT / 'src/models/graph/graph_wavenet.py',
        REPO_ROOT / 'src/training/train_graph_experts.py',
        REPO_ROOT / 'src/models/ensemble/hsdt.py',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete Tier-A package: {missing}")
    suffix = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:10]
    p104 = _load_module(f"tiera_p104_{suffix}", REPO_ROOT / 'src/training/train_graph_experts.py')
    p108 = _load_module(f"tiera_p108_{suffix}", REPO_ROOT / 'src/models/ensemble/hsdt.py')
    return argparse.Namespace(root=root, p104=p104, p108=p108, final4=p104.final4, v2=p104.v2, priority1=p104.priority1)


def validate_protocol(tiera_root: Path, contract: TierAContract = TierAContract()) -> dict[str, Any]:
    protocol_path = Path(tiera_root).resolve() / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    training = protocol["training"]
    checks = {
        "task": protocol["task"] == "past_24h_to_future_24h_non_tidal_residual",
        "seeds": tuple(training["seeds"]) == SEEDS,
        "train_stride": training["train_stride"] == contract.train_stride,
        "hidden_dim": training["hidden_dim"] == contract.hidden_dim,
        "blocks": training["gwn_blocks"] == contract.gwn_blocks,
        "diffusion": training["diffusion_steps"] == contract.diffusion_steps,
        "dropout": np.isclose(training["dropout"], contract.dropout),
        "batch": training["batch_size"] == contract.batch_size,
        "patience": training["patience"] == contract.patience,
        "lr": np.isclose(training["learning_rate"], contract.learning_rate),
        "weight_decay": np.isclose(training["weight_decay"], contract.weight_decay),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"Tier-A protocol mismatch in {protocol_path}: {failed}")
    return protocol


def make_args(contract: TierAContract, **overrides) -> argparse.Namespace:
    values = {
        "window": contract.input_hours,
        "horizon": contract.forecast_hours,
        "train_ratio": contract.train_ratio,
        "val_ratio": contract.val_ratio,
        "train_stride": contract.train_stride,
        "fixed_graph_type": contract.fixed_graph_type,
        "hidden_dim": contract.hidden_dim,
        "diffusion_steps": contract.diffusion_steps,
        "gwn_blocks": contract.gwn_blocks,
        "dropout": contract.dropout,
        "batch_size": contract.batch_size,
        "epochs": contract.epochs,
        "patience": contract.patience,
        "min_delta": 1e-5,
        "lr": contract.learning_rate,
        "weight_decay": contract.weight_decay,
        "grad_clip": contract.grad_clip,
        "aux_weight": contract.aux_weight,
        "last_step_weight": contract.last_step_weight,
        "physics_lambda": contract.physics_lambda,
        "physics_warmup_epochs": contract.physics_warmup_epochs,
        "physics_ramp_epochs": contract.physics_ramp_epochs,
        "physics_lr_mult": 0.5,
        "ode_coef_l2": 1e-5,
        "extreme_quantile": 0.90,
        "event_quantile": 0.95,
        "physics_forcing_mode": "last_input",
        "print_every": 5,
        "epoch_checkpoint_every": 1,
        "cpu_threads": 4,
        "resume": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def set_reproducible(seed: int, cpu_threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_formal_data(tiera_root: Path, contract: TierAContract = TierAContract()):
    validate_protocol(tiera_root, contract)
    modules = load_official_modules(str(Path(tiera_root).resolve()))
    args = make_args(contract)
    data = modules.final4.build_enhanced_data(args, contract.forecast_hours, add_ode_prior=False)
    if data["feats"] != contract.n_features:
        raise RuntimeError(f"Expected {contract.n_features} formal features, got {data['feats']}")
    if len(data["single_test"]) != contract.test_windows:
        raise RuntimeError(f"Expected {contract.test_windows} test windows, got {len(data['single_test'])}")
    if data["feature_cols"][-1] == "ode_local_trend_3h":
        raise RuntimeError("ODE-prior features leaked into the locked 34-feature contract")
    return modules, args, data


class StationSliceSingle(Dataset):
    def __init__(self, base: Dataset, station_index: int):
        self.base = base
        self.station_index = int(station_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y, tide = self.base[index]
        i = self.station_index
        return x[:, i : i + 1, :], y[i : i + 1, :], tide[i : i + 1, :]


class StationSliceMulti(Dataset):
    def __init__(self, base: Dataset, station_index: int):
        self.base = base
        self.station_index = int(station_index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, states, tide, initial, forcing = self.base[index]
        i = self.station_index
        return (
            x[:, i : i + 1, :],
            states[i : i + 1, :, :],
            tide[i : i + 1, :],
            initial[i : i + 1, :],
            forcing[i : i + 1, :, :],
        )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def extract_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "best_state_dict" in payload and payload["best_state_dict"] is not None:
        state = payload["best_state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        return state
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError(f"No model state found in {path}")


def locate_formal_checkpoint(results_root: Path, seed: int, config: str) -> Path:
    run_dir = Path(results_root) / f"seed_{seed}" / "horizon_24h" / config
    candidates = [run_dir / "best_checkpoint.pt", run_dir / "last_epoch_checkpoint.pt"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing formal checkpoint for seed={seed}, config={config}: {candidates}")


def write_run_manifest(
    output_dir: Path,
    *,
    tiera_root: Path,
    contract: TierAContract,
    feature_cols: list[str],
    station_ids: list[str],
    extra: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tiera_root).resolve()
    scripts = root
    payload = {
        "contract": asdict(contract),
        "feature_cols": list(feature_cols),
        "station_ids": list(station_ids),
        "protocol_sha256": sha256_file(root / "PROTOCOL.json"),
        "source_script_hashes": {
            name: sha256_file(scripts / name)
            for name in (
                "src/models/graph/graph_wavenet.py",
                "src/training/train_graph_experts.py",
                "src/models/ensemble/hsdt.py",
            )
        },
        "evidence_boundary": "retrospective_aligned_forcing_tier_a",
        **extra,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
