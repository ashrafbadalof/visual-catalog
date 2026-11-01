import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

def softmax(logits, T=1.0):
    z = logits / T
    z = z - z.max(axis=1, keepdims = True)
    e = np.exp(z)
    return e / e.sum(axis = 1, keepdims = True)

def topk_accuracy(logits, y, k):
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())

def expected_calibration_error(probs, y, n_bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = pred == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def classification_report(logits, y, class_names, top_confusions=15):
    C = len(class_names)
    pred = logits.argmax(axis=1)
    probs = softmax(logits)

    summary = {
        "n": int(len(y)),
        "top1": float((pred == y).mean()),
        "top5": topk_accuracy(logits, y, 5),
        "macro_f1": float(f1_score(y, pred, labels=range(C),
                                   average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, y),
    }

    cm = confusion_matrix(y, pred, labels=range(C))
    correct = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = correct / cm.sum(axis=1)
        precision = correct / cm.sum(axis=0)
    per_class = pd.DataFrame({
        "class": class_names,
        "support": cm.sum(axis=1),
        "recall": recall,
        "precision": precision,
    }).sort_values("recall")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
    confusions = [
        {"true": class_names[t], "predicted": class_names[p],
         "count": int(off[t, p]), "share_of_true": float(off[t, p] / cm[t].sum())}
        for t, p in pairs[:top_confusions] if off[t, p] > 0
    ]
    return summary, per_class, confusions