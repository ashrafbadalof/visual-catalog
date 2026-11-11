import argparse

from catalog.config import data_path, load_config
from catalog.encoders.onnx_student import OnnxStudentEncoder
from catalog.store.ingest import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="teacher", choices=["teacher", "student"])
    cfg = load_config()
    encoder = None
    if p.parse_args().encoder == "student":
        run_name = cfg["cascade"]["student_run"]
        encoder = OnnxStudentEncoder(data_path(cfg, "checkpoints_dir") / f"{run_name}.onnx",
                                     f"student-onnx/{run_name}")
    run(cfg, encoder)