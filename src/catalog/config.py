import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "configs" / "base.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = Path(os.environ.get("CATALOG_DATA_ROOT", cfg["data_root"]))
    cfg["data_root"] = root if root.is_absolute() else REPO_ROOT / root
    return cfg

def data_path(cfg, key):
    return cfg["data_root"] / cfg["paths"][key]