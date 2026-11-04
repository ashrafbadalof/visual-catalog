import argparse
import json

from catalog.config import REPO_ROOT, data_path, load_config
from catalog.encoders.teacher import ClipTeacher
from catalog.service.app import build_app
from catalog.service.tagger import Tagger
from catalog.store.sqlite_store import SQLiteStore

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    device = p.parse_args().device

    cfg = load_config()
    t = cfg["teacher"]
    classes = json.loads((data_path(cfg, "processed_dir") / "classes.json").read_text())

    teacher = ClipTeacher(t["arch"], t["pretrained"], list(classes.values()),
                          t["prompts"], device=device)
    store = SQLiteStore(data_path(cfg, "catalog_db"))
    tagger = Tagger(teacher, store, list(classes), cfg["data_root"],
                    REPO_ROOT / "results" / "latency.csv")
    build_app(tagger).launch(allowed_paths=[str(cfg["data_root"])])