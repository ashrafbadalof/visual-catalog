import json

import pandas as pd
from sklearn.model_selection import train_test_split

from catalog.config import data_path

KEEP_COLUMNS = ["id", "path", "label", "split", "productDisplayName",
                "gender", "masterCategory", "subCategory", "baseColour"]

def load_styles(csv_path, images_dir):
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df.dropna(subset=["id", "articleType", "productDisplayName"])
    df["id"] = df["id"].astype(int)
    df["path"] = df["id"].astype(str) + ".jpg"  # relative to images_dir
    has_image = df["path"].map(lambda p: (images_dir / p).exists())
    print(f"Dropped {(~has_image).sum()} rows with no image")
    return df[has_image].reset_index(drop=True)


def choose_classes(df, column, min_images):
    counts = df[column].value_counts()
    keep = counts[counts >= min_images].index
    coverage = counts[keep].sum() / len(df)
    print(f"Kept {len(keep)} of {len(counts)} classes, covering {coverage:.1%} of images")
    df = df[df[column].isin(keep)].copy()
    df["label"] = df[column]
    return df

def make_splits(df, val_frac, test_frac, seed):
    rest, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        rest, test_size=val_frac / (1 - test_frac),
        stratify=rest["label"], random_state=seed)
    return pd.concat([train.assign(split="train"),
                      val.assign(split="val"),
                      test.assign(split="test")], ignore_index=True)

def write_classes(labels, path):
    existing = json.loads(path.read_text()) if path.exists() else {}
    classes = {c: existing.get(c, c.lower()) for c in sorted(labels)}
    path.write_text(json.dumps(classes, indent=2))


def run(cfg):
    images_dir = data_path(cfg, "images_dir")
    out_dir = data_path(cfg, "processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_styles(data_path(cfg, "styles_csv"), images_dir)
    df = choose_classes(df, cfg["labels"]["column"], cfg["labels"]["min_images"])
    df = make_splits(df, cfg["splits"]["val"], cfg["splits"]["test"], cfg["seed"])

    df[KEEP_COLUMNS].to_parquet(out_dir / "splits.parquet", index=False)
    write_classes(df["label"].unique(), out_dir / "classes.json")

    test = df[df["split"] == "test"]
    test[["id", "productDisplayName"]].rename(
        columns={"productDisplayName": "query"}
    ).to_parquet(out_dir / "queries.parquet", index=False)

    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))