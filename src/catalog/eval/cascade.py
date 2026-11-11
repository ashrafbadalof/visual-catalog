import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from catalog.config import REPO_ROOT, data_path
from catalog.encoders.student import load_student
from catalog.eval.classification import softmax
from catalog.train.head import train_head

DEFER_RATES = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


@torch.no_grad()
def zero_shot_logits(cfg, c, feats):
    student = load_student(data_path(cfg, "checkpoints_dir") / f"{c['student_run']}.pt",
                           data_path(cfg, "cache_dir"))
    emb = F.normalize(student.net.classifier[-1](feats), dim=-1)
    return (student.logit_scale * emb @ student.class_embs.T).numpy()


def soft_head_logits(feats, teacher_logits, is_train, T):
    X = (feats - feats[is_train].mean(0)) / (feats[is_train].std(0) + 1e-6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    targets = F.softmax(teacher_logits[is_train] / T, dim=1).to(device)
    head = train_head(X[is_train.to(device)], targets, T)
    return head(X).detach().cpu().numpy()


def run(cfg):
    c = cfg["cascade"]
    cache = data_path(cfg, "cache_dir")
    df = pd.read_parquet(data_path(cfg, "processed_dir") / "splits.parquet")
    classes = json.loads((cache / "metadata.json").read_text())["classes"]
    split = df["split"].to_numpy()
    val, test = split == "val", split == "test"
    y = df["label"].map({c_: i for i, c_ in enumerate(classes)}).to_numpy()

    teacher_logits = torch.from_numpy(np.load(cache / "logits.npy"))
    teacher_pred = teacher_logits.argmax(1).numpy()
    feats = torch.from_numpy(
        np.load(data_path(cfg, "features_dir") / f"{c['student_run']}.npy").astype(np.float32))

    routes = {
        "zero_shot": zero_shot_logits(cfg, c, feats),
        "soft_head": soft_head_logits(feats, teacher_logits,
                                      torch.from_numpy(split == "train"), c["head_T"]),
    }

    rows = []
    for name, logits in routes.items():
        conf, pred = softmax(logits).max(1), logits.argmax(1)
        for rate in DEFER_RATES:
            threshold = (-np.inf if rate == 0 else np.inf if rate == 1
                         else np.quantile(conf[val], rate))
            defer = conf[test] < threshold
            final = np.where(defer, teacher_pred[test], pred[test])
            rows.append({
                "route": name,
                "target_defer": rate,
                "threshold": round(float(threshold), 4) if np.isfinite(threshold) else None,
                "defer_rate": round(defer.mean(), 4),
                "top1": round((final == y[test]).mean(), 4),
                "macro_f1": round(f1_score(y[test], final, average="macro"), 4),
                "ms_per_image": round(c["student_ms"] + defer.mean() * c["teacher_ms"], 1),
            })

    results = pd.DataFrame(rows)
    results.to_csv(REPO_ROOT / "results" / "stage4_cascade.csv", index=False)
    print(results.to_string(index=False))