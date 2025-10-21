from __future__ import annotations

import argparse
import importlib.util
import sys
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = config.load_config(config_path)
    trainer = train_mod.FMIntegrator(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
