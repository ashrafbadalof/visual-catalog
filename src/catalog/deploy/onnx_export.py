import json
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from catalog.config import REPO_ROOT, data_path
from catalog.encoders.student import load_student


class ExportWrapper(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.net = student.net
        self.register_buffer("class_embs", student.class_embs)
        self.logit_scale = student.logit_scale

    def forward(self, x):
        emb = F.normalize(self.net(x), dim=-1)
        return emb, self.logit_scale * emb @ self.class_embs.T


def median_ms(fn, inputs, warmup=5):
    times = []
    for i, x in enumerate(inputs):
        start = time.perf_counter()
        fn(x)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def run(cfg):
    run_name = cfg["cascade"]["student_run"]
    ckpt_dir = data_path(cfg, "checkpoints_dir")
    student = load_student(ckpt_dir / f"{run_name}.pt", data_path(cfg, "cache_dir"), device="cpu")
    model = ExportWrapper(student).eval()

    # 1. Export
    onnx_path = ckpt_dir / f"{run_name}.onnx"
    torch.onnx.export(
        model, torch.randn(1, 3, 224, 224), onnx_path,
        input_names=["image"], output_names=["embedding", "logits"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False)

    df = pd.read_parquet(data_path(cfg, "processed_dir") / "splits.parquet")
    images_dir = data_path(cfg, "images_dir")

    def load(path):
        with Image.open(images_dir / path) as im:
            return student.preprocess(im.convert("RGB")).unsqueeze(0)

    inputs = [load(p) for p in df.loc[df["split"] == "test", "path"][:55]]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    batch = torch.cat(inputs[:10])
    with torch.no_grad():
        torch_emb, torch_logits = model(batch)
    onnx_emb, onnx_logits = session.run(None, {"image": batch.numpy()})

    with torch.no_grad():
        torch_ms = median_ms(model, inputs)
    onnx_ms = median_ms(lambda x: session.run(None, {"image": x.numpy()}), inputs)

    results = {
        "student_run": run_name,
        "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "min_cosine_onnx_vs_torch": float((torch_emb.numpy() * onnx_emb).sum(axis=1).min()),
        "same_predictions": bool((torch_logits.argmax(1).numpy() == onnx_logits.argmax(1)).all()),
        "torch_cpu_ms": round(torch_ms, 2),
        "onnx_cpu_ms": round(onnx_ms, 2),
    }
    (REPO_ROOT / "results" / f"onnx_{run_name}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))