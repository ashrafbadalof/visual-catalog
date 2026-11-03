import json
from datetime import datetime

import numpy as np
import pandas as pd

from catalog.config import data_path
from catalog.encoders.teacher import ClipTeacher


def run(cfg):
    processed = data_path(cfg, "processed_dir")
    cache = data_path(cfg, "cache_dir")
    t = cfg["teacher"]

    df = pd.read_parquet(processed / "splits.parquet")
    classes = json.loads((processed / "classes.json").read_text())
    meta = json.loads((cache / "metadata.json").read_text())

    assert meta["teacher"] == f"{t['arch']}/{t['pretrained']}", \
        "Teacher model changed: run cache_teacher.py instead"

    old_ids = np.load(cache / "ids.npy")
    position = pd.Series(np.arange(len(old_ids)), index=old_ids)
    missing = ~df["id"].isin(position.index)
    assert not missing.any(), f"{missing.sum()} images not in cache: run cache_teacher.py"
    embs = np.load(cache / "image_embs.npy")[position.loc[df["id"]].to_numpy()]

    teacher = ClipTeacher(t["arch"], t["pretrained"], list(classes.values()), t["prompts"])
    class_embs = teacher.class_embs.cpu().numpy()
    logits = teacher.logit_scale * embs.astype(np.float32) @ class_embs.T

    queries = pd.read_parquet(processed / "queries.parquet")
    query_embs = teacher.embed_texts(queries["query"].tolist())

    np.save(cache / "image_embs.npy", embs)
    np.save(cache / "ids.npy", df["id"].to_numpy())
    np.save(cache / "logits.npy", logits.astype(np.float32))
    np.save(cache / "class_embs.npy", class_embs)
    np.save(cache / "query_embs.npy", query_embs.astype(np.float16))
    np.save(cache / "query_ids.npy", queries["id"].to_numpy())

    meta.update({
        "prompts": t["prompts"],
        "classes": list(classes),
        "num_images": len(df),
        "relabelled": datetime.now().isoformat(timespec="seconds"),
    })
    (cache / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Relabelled {len(df)} images with {len(classes)} classes (no image re-encoding)")