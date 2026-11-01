import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from catalog.config import data_path
from catalog.data.dataset import ImageDataset
from catalog.encoders.teacher import ClipTeacher

def run(cfg):
    processed = data_path(cfg, "processed_dir")
    out = data_path(cfg, "cache_dir")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed / "splits.parquet")
    classes = json.loads((processed / "classes.json").read_text())
    t = cfg["teacher"]

    teacher = ClipTeacher(t["arch"], t["pretrained"],
                          list(classes.values()), t["prompts"])

    ds = ImageDataset(df["path"], data_path(cfg, "images_dir"), teacher.preprocess)
    loader = DataLoader(ds, batch_size=t["batch_size"],
                        num_workers=t["num_workers"])

    start = time.time()
    embs, logits = [], []
    for batch in tqdm(loader, desc="Teacher"):
        e, l = teacher.encode_images(batch)
        embs.append(e.astype(np.float16))
        logits.append(l.astype(np.float32))
    seconds = time.time() - start

    np.save(out / "image_embs.npy", np.concatenate(embs))
    np.save(out / "logits.npy", np.concatenate(logits))
    np.save(out / "ids.npy", df["id"].to_numpy())
    np.save(out / "class_embs.npy", teacher.class_embs.cpu().numpy())

    queries = pd.read_parquet(processed / "queries.parquet")
    np.save(out / "query_embs.npy",
            teacher.embed_texts(queries["query"].tolist()).astype(np.float16))
    np.save(out / "query_ids.npy", queries["id"].to_numpy())

    meta = {
        "teacher": teacher.name,
        "prompts": t["prompts"],
        "classes": list(classes),
        "logit_scale": teacher.logit_scale,
        "num_images": len(df),
        "seconds": round(seconds, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Cached {len(df)} images in {seconds / 60:.1f} min → {out}")