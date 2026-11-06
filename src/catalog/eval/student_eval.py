import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from catalog.config import REPO_ROOT, data_path
from catalog.data.dataset import ImageDataset
from catalog.encoders.student import load_student
from catalog.eval.classification import classification_report
from catalog.eval.retrieval import text_to_image_retrieval


def cpu_latency_ms(encoder, files, warmup=5):
    times = []
    for i, f in enumerate(files):
        with Image.open(f) as im:
            image = im.convert("RGB")
        start = time.perf_counter()
        encoder.encode_images(encoder.preprocess(image).unsqueeze(0))
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def run(cfg):
    cache = data_path(cfg, "cache_dir")
    processed = data_path(cfg, "processed_dir")
    images_dir = data_path(cfg, "images_dir")
    run_name = cfg["student"]["run_name"]
    checkpoint = data_path(cfg, "checkpoints_dir") / f"{run_name}.pt"

    df = pd.read_parquet(processed / "splits.parquet")
    meta = json.loads((cache / "metadata.json").read_text())
    test = (df["split"] == "test").to_numpy()
    rows = df[test]

    student = load_student(checkpoint, cache, device="cuda")
    loader = DataLoader(ImageDataset(rows["path"], images_dir, student.preprocess),
                        batch_size=64, num_workers=4)
    embs, logits = [], []
    for batch in loader:
        e, l = student.encode_images(batch)
        embs.append(e)
        logits.append(l)
    embs, logits = np.concatenate(embs), np.concatenate(logits)

    teacher = np.load(cache / "image_embs.npy")[test].astype(np.float32)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    y = rows["label"].map({c: i for i, c in enumerate(meta["classes"])}).to_numpy()
    queries = pd.read_parquet(processed / "queries.parquet")

    cpu_student = load_student(checkpoint, cache, device="cpu")
    results = {
        "run_name": run_name,
        "student": student.name,
        "params_millions": round(sum(p.numel() for p in student.net.parameters()) / 1e6, 2),
        "agreement_with_teacher": float((embs * teacher).sum(axis=1).mean()),
        "classification": classification_report(logits, y, meta["classes"])[0],
        "retrieval": text_to_image_retrieval(
            np.load(cache / "query_embs.npy").astype(np.float32),
            np.load(cache / "query_ids.npy"), queries["query"].to_numpy(),
            embs, rows["id"].to_numpy(), rows["productDisplayName"].to_numpy()),
        "cpu_ms_per_image": cpu_latency_ms(cpu_student, [images_dir / p for p in rows["path"][:55]]),
    }
    out = REPO_ROOT / "results" / f"student_{run_name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))