import json

import pandas as pd

from catalog.config import load_config, data_path

cfg = load_config()
processed = data_path(cfg, "processed_dir")
df = pd.read_parquet(processed / "splits.parquet")


def test_no_id_in_two_splits():
    assert df.groupby("id")["split"].nunique().max() == 1


def test_all_images_exist():
    images_dir = data_path(cfg, "images_dir")
    assert all((images_dir / p).exists() for p in df["path"])


def test_every_label_has_a_class_entry():
    classes = json.loads((processed / "classes.json").read_text())
    assert set(df["label"]) == set(classes)


def test_every_class_in_every_split():
    assert (df.groupby(["label", "split"]).size().unstack(fill_value=0) > 0).all().all()