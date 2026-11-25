"""Build the static, in-browser demo: models, catalog and page in one folder."""
import json
import shutil

import numpy as np
import onnxruntime as ort
import open_clip
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic

from catalog.config import REPO_ROOT, data_path
from catalog.eval.retrieval import text_to_image_retrieval
from catalog.store.sqlite_store import SQLiteStore

# Same BPE vocabulary as open_clip's tokenizer, in the format transformers.js reads.
TOKENIZER_REPO = "openai/clip-vit-large-patch14"


class TextTower(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, tokens):
        emb = self.clip.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)


def export_text_tower(cfg, models_dir):
    t = cfg["teacher"]
    clip, _, _ = open_clip.create_model_and_transforms(t["arch"], pretrained=t["pretrained"])
    tokens = open_clip.get_tokenizer(t["arch"])(["a photo of a shirt"])
    fp32 = models_dir / "text_fp32.onnx"
    torch.onnx.export(
        TextTower(clip).eval(), tokens, fp32,
        input_names=["tokens"], output_names=["embedding"],
        dynamic_axes={"tokens": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False)
    quantize_dynamic(fp32, models_dir / "text_int8.onnx", weight_type=QuantType.QInt8)
    fp32.unlink()

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(hf_hub_download(TOKENIZER_REPO, name), models_dir.parent / "tokenizer" / name)


def export_catalog(cfg, out):
    db = SQLiteStore(data_path(cfg, "catalog_db")).db
    rows = db.execute("""SELECT p.id, p.image_path, p.title, p.category, v.embedding
                         FROM products p JOIN product_vectors v ON v.rowid = p.id
                         WHERE p.image_path LIKE 'images_256/%' ORDER BY p.id""").fetchall()
    embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    products = [{"id": r["id"], "image": r["image_path"], "title": r["title"], "category": r["category"]}
                for r in rows]

    (out / "catalog" / "embeddings.bin").write_bytes(embs.tobytes())
    (out / "catalog" / "products.json").write_text(json.dumps(products))
    classes = json.loads((data_path(cfg, "processed_dir") / "classes.json").read_text())
    (out / "catalog" / "classes.json").write_text(json.dumps(list(classes)))

    for p in products:
        dst = out / p["image"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg["data_root"] / p["image"], dst)
    return products, embs


def check_search(cfg, models_dir, products, embs):
    """Recall of the int8 text tower vs the original, on the same queries and catalog."""
    queries = pd.read_parquet(data_path(cfg, "processed_dir") / "queries.parquet")
    tokenizer = open_clip.get_tokenizer(cfg["teacher"]["arch"])
    session = ort.InferenceSession(str(models_dir / "text_int8.onnx"), providers=["CPUExecutionProvider"])
    texts = queries["query"].tolist()
    int8 = np.concatenate([session.run(None, {"tokens": tokenizer(texts[i:i + 256]).numpy()})[0]
                           for i in range(0, len(texts), 256)])
    cache = data_path(cfg, "cache_dir")
    original = np.load(cache / "query_embs.npy").astype(np.float32)

    ids = np.array([p["id"] for p in products])
    titles = np.array([p["title"] for p in products])
    keep = ["exact_recall@1", "exact_recall@10", "exact_mrr"]
    results = {}
    for name, q, q_ids in (("original", original, np.load(cache / "query_ids.npy")),
                           ("int8", int8, queries["id"].to_numpy())):
        r = text_to_image_retrieval(q, q_ids, queries["query"].to_numpy(), embs, ids, titles)
        results[name] = {k: round(r[k], 4) for k in keep}
    results["text_model_mb"] = round((models_dir / "text_int8.onnx").stat().st_size / 1e6, 1)
    return results


def build(cfg, out):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO_ROOT / "deploy" / "web", out)
    for d in ("models", "tokenizer", "catalog"):
        (out / d).mkdir()

    run = cfg["cascade"]["student_run"]
    shutil.copy2(data_path(cfg, "checkpoints_dir") / f"{run}.onnx", out / "models" / "student.onnx")
    export_text_tower(cfg, out / "models")
    products, embs = export_catalog(cfg, out)

    results = check_search(cfg, out / "models", products, embs)
    (REPO_ROOT / "results" / "web_search_check.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
