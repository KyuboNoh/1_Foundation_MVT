from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent


def _import_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = _import_local("integrate_config", CURRENT_DIR / "config.py")
sys.modules.setdefault("config", config)
data_mod = _import_local("integrate_data", CURRENT_DIR / "data.py")
sys.modules.setdefault("data", data_mod)
losses_mod = _import_local("integrate_losses", CURRENT_DIR / "losses.py")
sys.modules.setdefault("losses", losses_mod)
models_mod = _import_local("integrate_models", CURRENT_DIR / "models.py")
sys.modules.setdefault("models", models_mod)
train_mod = _import_local("integrate_train", CURRENT_DIR / "train.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate multiple foundation models across datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug diagnostics regardless of config.")
    parser.add_argument(
        "--debug-log-every",
        type=int,
        default=None,
        help="Override debug logging frequency (epochs). Requires --debug to take effect.",
    )
    parser.add_argument("--inference", action="store_true", help="Generate inference maps after training.")
    parser.add_argument(
        "--use-previous-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pre-defined PN splits for negative sampling (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = config.load_config(config_path)
    if args.debug:
        cfg.debug = True
        if args.debug_log_every is not None:
            cfg.debug_log_every = max(1, args.debug_log_every)
    if args.inference:
        cfg.generate_inference = True
    cfg.use_previous_negatives = args.use_previous_negatives
    print(f"[info] Loaded integration config from {config_path}")
    if hasattr(cfg, "loss_weights"):
        weights = asdict(cfg.loss_weights)
        weights_str = ", ".join(f"{key}={value}" for key, value in weights.items())
        print(f"[info] Active loss weights -> {weights_str}")
    if cfg.debug:
        print(f"[info] Debug mode enabled; logging every {cfg.debug_log_every} epoch(s)")
    if getattr(cfg, "generate_inference", False):
        out_dir = cfg.output_dir / "inference"
        print(f"[info] Inference generation enabled; outputs will be written under {out_dir}")
    trainer = train_mod.FMIntegrator(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
