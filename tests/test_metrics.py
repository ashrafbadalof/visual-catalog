import numpy as np

from catalog.eval.classification import classification_report
from catalog.eval.retrieval import text_to_image_retrieval


def test_perfect_classifier():
    y = np.array([0, 1, 2, 1])
    logits = np.eye(3)[y] * 10
    summary, _, confusions = classification_report(logits, y, ["a", "b", "c"])
    assert summary["top1"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert confusions == []


def test_perfect_retrieval():
    embs = np.eye(4)
    ids = np.array([10, 11, 12, 13])
    titles = np.array(["w", "x", "y", "z"])
    r = text_to_image_retrieval(embs, ids, titles, embs, ids, titles)
    assert r["exact_recall@1"] == 1.0
    assert r["exact_mrr"] == 1.0