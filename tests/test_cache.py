import numpy as np
import pandas as pd

from catalog.config import load_config, data_path

cfg = load_config()
cache = data_path(cfg, "cache_dir")
df = pd.read_parquet(data_path(cfg, "processed_dir") / "splits.parquet")

def test_rows_match_splits():
    assert len(np.load(cache / "image_embs.npy")) == len(df)
    assert len(np.load(cache / "logits.npy")) == len(df)


def test_ids_in_same_order():
    assert (np.load(cache / "ids.npy") == df["id"].to_numpy()).all()


def test_embeddings_normalised():
    norms = np.linalg.norm(np.load(cache / "image_embs.npy").astype(np.float32), axis=1)
    assert np.allclose(norms, 1, atol=1e-2)