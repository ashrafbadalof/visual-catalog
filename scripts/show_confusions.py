import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

from catalog.config import REPO_ROOT, data_path, load_config


def make_grid(paths, images_dir, cols=5, size=200):
    rows = max(1, -(-len(paths) // cols))  # ceiling division
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    for i, p in enumerate(paths):
        with Image.open(images_dir / p) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            x = (i % cols) * size + (size - im.width) // 2
            y = (i // cols) * size + (size - im.height) // 2
            sheet.paste(im, (x, y))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", required=True, help="human label, e.g. Heels")
    ap.add_argument("--pred", required=True, help="teacher prediction, e.g. Sandals")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    cfg = load_config()
    cache = data_path(cfg, "cache_dir")
    images_dir = data_path(cfg, "images_dir")
    df = pd.read_parquet(data_path(cfg, "processed_dir") / "splits.parquet")
    classes = json.loads((cache / "metadata.json").read_text())["classes"]
    df["pred"] = np.array(classes)[np.load(cache / "logits.npy").argmax(axis=1)]

    rows = df[df["split"] == a.split]
    wrong = rows[(rows["label"] == a.true) & (rows["pred"] == a.pred)]
    reference = rows[(rows["label"] == a.pred) & (rows["pred"] == a.pred)]

    out = REPO_ROOT / "results" / "confusions"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.true}_as_{a.pred}".replace(" ", "_")
    make_grid(wrong["path"].head(a.n).tolist(), images_dir).save(out / f"{name}.jpg")
    make_grid(reference["path"].head(a.n).tolist(), images_dir).save(out / f"{name}_reference.jpg")

    print(f"{len(wrong)} images labelled '{a.true}' were predicted '{a.pred}'")
    print(wrong[["id", "productDisplayName"]].head(a.n).to_string(index=False))


if __name__ == "__main__":
    main()