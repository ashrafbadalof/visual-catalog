import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from catalog.config import REPO_ROOT, data_path
from catalog.eval.classification import classification_report
from catalog.train.head import train_head

WATCH = ["Heels", "Wallets", "Casual Shoes"]


def run(cfg):
    h = cfg["head"]
    cache, feats_dir = data_path(cfg, "cache_dir"), data_path(cfg, "features_dir")
    df = pd.read_parquet(data_path(cfg, "processed_dir") / "splits.parquet")
    classes = json.loads((cache / "metadata.json").read_text())["classes"]
    assert (np.load(feats_dir / "ids.npy") == df["id"].to_numpy()).all(), "Features out of sync"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(np.load(feats_dir / f"{h['student_run']}.npy"), dtype=torch.float32)
    split = df["split"].to_numpy()
    is_train = torch.from_numpy(split == "train")
    X = ((X - X[is_train].mean(0)) / (X[is_train].std(0) + 1e-6)).to(device)
    teacher = torch.tensor(np.load(cache / "logits.npy"), device=device)
    y = torch.tensor(df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy(), device=device)
    idx = {s: torch.tensor(np.flatnonzero(split == s), device=device) for s in ("train", "val", "test")}
    train, val, test = idx["train"], idx["val"], idx["test"]

    def make_targets(kind, rows, T):
        if kind == "soft":
            return F.softmax(teacher[rows] / T, dim=1)
        labels = teacher[rows].argmax(1) if kind == "hard" else y[rows]
        return F.one_hot(labels, len(classes)).float()

    def score(logits, rows):
        summary, per_class, _ = classification_report(
            logits.detach().cpu().numpy(), y[rows].cpu().numpy(), classes)
        recall = per_class.set_index("class")["recall"]
        return {"top1": summary["top1"], "macro_f1": summary["macro_f1"], "ece": summary["ece"],
                **{f"recall_{c}": float(recall[c]) for c in WATCH}}

    val_top1 = {T: score(train_head(X[train], make_targets("soft", train, T), T,
                                    steps=h["steps"])(X[val]), val)["top1"]
                for T in h["temperatures"]}
    T = max(val_top1, key=val_top1.get)
    print(f"Val top-1 by temperature: {val_top1} → using T = {T}")

    rows = [{"experiment": "reference", "kind": "teacher", **score(teacher[test], test)}]

    for n in h["budgets"]:
        for seed in range(h["seeds"] if n < len(train) else 1):
            g = torch.Generator().manual_seed(seed)
            subset = train[torch.randperm(len(train), generator=g)[:n].to(device)]
            for kind in ("soft", "hard", "human"):
                t = T if kind == "soft" else 1.0
                head = train_head(X[subset], make_targets(kind, subset, t), t,
                                  steps=h["steps"], seed=seed)
                rows.append({"experiment": "budget", "kind": kind, "n": len(subset),
                             "seed": seed, **score(head(X[test]), test)})

    train_y = y[train].cpu().numpy()
    for k in h["human_per_class"]:
        for seed in range(h["seeds"]):
            rng = np.random.default_rng(seed)
            picked = np.concatenate([rng.choice(np.flatnonzero(train_y == c),
                                                min(k, (train_y == c).sum()), replace=False)
                                     for c in range(len(classes))])
            human = train[torch.from_numpy(picked).to(device)]
            head = train_head(X[train], make_targets("soft", train, T), T,
                              X_h=X[human], y_h=y[human], steps=h["steps"], seed=seed)
            rows.append({"experiment": "ceiling", "kind": f"soft+{k}/class", "n": len(train),
                         "seed": seed, **score(head(X[test]), test)})

    results = pd.DataFrame(rows)
    results.to_csv(REPO_ROOT / "results" / "stage3_heads.csv", index=False)
    print(results.groupby(["experiment", "kind", "n"], dropna=False)[["top1", "macro_f1"]]
          .agg(["mean", "std"]).round(4).to_string())