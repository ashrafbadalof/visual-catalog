import json

import numpy as np
import pandas as pd

from catalog.config import REPO_ROOT, data_path
from catalog.eval.classification import classification_report
from catalog.eval.retrieval import text_to_image_retrieval


def run(cfg, split):
    processed = data_path(cfg, "processed_dir")
    cache = data_path(cfg, "cache_dir")
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)

    df = pd.read_parquet(processed / "splits.parquet")
    meta = json.loads((cache / "metadata.json").read_text())
    class_names = meta["classes"]
    assert (np.load(cache / "ids.npy") == df["id"].to_numpy()).all(), "Cache out of sync"

    mask = (df["split"] == split).to_numpy()
    logits = np.load(cache / "logits.npy")[mask]
    y = df.loc[mask, "label"].map({c: i for i, c in enumerate(class_names)}).to_numpy()

    summary, per_class, confusions = classification_report(logits, y, class_names)
    results = {"teacher": meta["teacher"], "prompts": meta["prompts"],
               "split": split, "classification": summary,
               "top_confusions": confusions}

    if split == "test":
        gallery = df[mask]
        queries = pd.read_parquet(processed / "queries.parquet")
        embs = np.load(cache / "image_embs.npy")[mask].astype(np.float32)
        results["retrieval"] = text_to_image_retrieval(
            np.load(cache / "query_embs.npy").astype(np.float32),
            np.load(cache / "query_ids.npy"),
            queries["query"].to_numpy(),
            embs, gallery["id"].to_numpy(), gallery["productDisplayName"].to_numpy())

    (out / f"teacher_{split}.json").write_text(json.dumps(results, indent=2))
    per_class.to_csv(out / f"teacher_{split}_per_class.csv", index=False)

    print(json.dumps({k: results[k] for k in ("classification", "retrieval")
                      if k in results}, indent=2))
    print("\nWorst 10 classes:\n", per_class.head(10).to_string(index=False))
    print("\nTop confusions:")
    for c in confusions[:10]:
        print(f"  {c['true']:>22} → {c['predicted']:<22} "
              f"{c['count']:>4}  ({c['share_of_true']:.0%} of {c['true']})")