import argparse
import json

from catalog.config import REPO_ROOT, data_path, load_config
from catalog.encoders.onnx_student import OnnxStudentEncoder
from catalog.encoders.teacher import ClipTeacher
from catalog.service.app import build_app
from catalog.service.tagger import Tagger
from catalog.store.sqlite_store import SQLiteStore

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="cascade", choices=["teacher", "cascade"])
    mode = p.parse_args().mode

    cfg = load_config()
    t, c = cfg["teacher"], cfg["cascade"]
    classes = json.loads((data_path(cfg, "processed_dir") / "classes.json").read_text())
    teacher = ClipTeacher(t["arch"], t["pretrained"], list(classes.values()), t["prompts"], device="cpu")
    store = SQLiteStore(data_path(cfg, "catalog_db"))
    log_path = REPO_ROOT / "results" / f"latency_{mode}.csv"

    if mode == "teacher":
        tagger = Tagger(teacher, store, list(classes), cfg["data_root"], log_path)
    else:
        student = OnnxStudentEncoder(data_path(cfg, "checkpoints_dir") / f"{c['student_run']}.onnx",
                                     f"student-onnx/{c['student_run']}", text_fn=teacher.embed_texts)
        tagger = Tagger(student, store, list(classes), cfg["data_root"], log_path,
                        fallback=teacher, threshold=c["threshold"])

    build_app(tagger).launch(allowed_paths=[str(cfg["data_root"])])