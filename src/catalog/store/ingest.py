import json

import numpy as np
import pandas as pd

from catalog.config import data_path
from catalog.eval.classification import softmax
from catalog.store.sqlite_store import Product, SQLiteStore

def run(cfg):
    processed = data_path(cfg, "processed_dir")
    cache = data_path(cfg, "cache_dir")
    db_path = data_path(cfg, "catalog_db")
    db_path.unlink(missing_ok=True)

    df = pd.read_parquet(processed / "splits.parquet")
    meta = json.loads((cache / "metadata.json").read_text())
    assert (np.load(cache / "ids.npy") == df["id"].to_numpy()).all(), "Cache out of sync"

    test = (df["split"] == "test").to_numpy()
    rows = df[test]
    embs = np.load(cache / "image_embs.npy")[test].astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    probs = softmax(np.load(cache / "logits.npy")[test])

    products = [Product(id=int(r.id), image_path=f"{cfg['paths']['images_dir']}/{r.path}", title=r.productDisplayName,
                        category=meta["classes"][p.argmax()], confidence=float(p.max()),
                        encoder=meta["teacher"])
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