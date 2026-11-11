import json

import numpy as np
import pandas as pd

from catalog.config import data_path
from catalog.eval.classification import softmax
from catalog.store.sqlite_store import Product, SQLiteStore

from torch.utils.data import DataLoader
from catalog.data.dataset import ImageDataset

def run(cfg, encoder = None):
    processed = data_path(cfg, "processed_dir")
    cache = data_path(cfg, "cache_dir")
    db_path = data_path(cfg, "catalog_db")
    db_path.unlink(missing_ok=True)

    df = pd.read_parquet(processed / "splits.parquet")
    meta = json.loads((cache / "metadata.json").read_text())
    assert (np.load(cache / "ids.npy") == df["id"].to_numpy()).all(), "Cache out of sync"

    test = (df["split"] == "test").to_numpy()
    rows = df[test]

    if encoder is None:
        embs = np.load(cache / "image_embs.npy")[test].astype(np.float32)
        logits = np.load(cache / "logits.npy")[test]
        name = meta["teacher"]
    else:
        loader = DataLoader(ImageDataset(rows["path"], data_path(cfg, "images_dir"), encoder.preprocess),
                            batch_size=64, num_workers=4)
        outputs = [encoder.encode_images(batch) for batch in loader]
        embs = np.concatenate([e for e, _ in outputs])
        logits = np.concatenate([l for _, l in outputs])
        name = encoder.name

    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    probs = softmax(logits)

    products = [Product(id=int(r.id), image_path=f"{cfg['paths']['images_dir']}/{r.path}", title=r.productDisplayName,
                        category=meta["classes"][p.argmax()], confidence=float(p.max()),
                        encoder=name)
                for r, p in zip(rows.itertuples(), probs)]

    store = SQLiteStore(db_path, dim=embs.shape[1])
    store.add(products, embs)
    print(f"{store.count()} products in {db_path}")

    queries = np.load(cache / "query_embs.npy")[:50].astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    matches = sum(np.allclose(np.sort(embs @ q)[::-1][:10],
                              [s for _, s in store.search(q)], atol=1e-5)
                  for q in queries)
    print(f"Top-10 scores match NumPy on {matches}/50 queries")
    store.close()