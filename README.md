# Distilled Visual Catalog Engine

Product-photo tagging and text search for online catalogues, distilled from a large
CLIP model into a small model that runs on a CPU.

A 5.2 M-parameter MobileNetV3 student (21 MB as ONNX) keeps **99.5% of the teacher's
tagging accuracy** and **87% of its search Recall@10** at **~63× lower CPU latency**.
A confidence cascade that sends ~9% of images to the teacher **slightly exceeds the
teacher's accuracy (86.0% vs 85.7%) at ~10× lower average cost**.

**Live demo:** https://huggingface.co/spaces/ashrafb04/visual-catalog (runs entirely in the browser:
student tagging and text search, without the cascade)

## Headline results

Test split (6,322 products, human labels). CPU latency is single-image on a laptop.

| System | Size | Top-1 | Search Recall@10 | CPU ms / image |
|---|---|---|---|---|
| Teacher: CLIP ViT-L-14 | ~1.7 GB | 85.7% | 69.9% | 1,240 |
| Student: MobileNetV3-Large | 21 MB | 85.3% | 61.1% | 19.7 (PyTorch) · 2.7 model-only (ONNX) |
| Cascade: student + teacher below 0.656 confidence | both | **86.0%** | 61.1% | ~131 average |

## How it works

```
Product images ──► Teacher (CLIP ViT-L-14), run once ──► cached embeddings + class logits
                                                               │
                        Student (MobileNetV3) trained to copy the cached embeddings
                                                               │
                        Exported to ONNX (21 MB, runs without PyTorch)
                                                               │
 Upload ──► student tags it ──► confidence < 0.656? ──► teacher re-tags it
                                                               │
 Catalog: SQLite + sqlite-vec ◄── search queries embedded by the teacher's text tower
```

- **Tagging** compares the student's embedding with the teacher's text embeddings of
  the class names (zero-shot), so the student needs no classification training.
- **Search** compares a query's text embedding with every product's image embedding.

Data: [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
(Kaggle, not included). 42,143 images, 51 categories, stratified 80/5/15 split.

## Stage 1: Teacher baseline (ceiling)

OpenCLIP ViT-L-14 (`laion2b_s32b_b82k`), zero-shot. Prompt names and class merges
(Kurtis/Tunics → Kurtas, Flats → Sandals) were tuned on validation only.

| Top-1 | Top-5 | Macro-F1 | ECE | Recall@1 / @10 (exact) | MRR | CPU tag / search |
|---|---|---|---|---|---|---|
| 85.7% | 99.7% | 0.861 | 0.051 | 27.6% / 69.9% | 0.413 | 1.24 s / 210 ms |

## Stage 2: Embedding students

Cosine loss against cached teacher embeddings, 20 epochs, batch 32.

| Model | Params | CPU ms | Agreement | Top-1 | Macro-F1 | Recall@1 | Recall@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Large | 5.2 M | 19.7 | 0.906 | 85.3% | 0.840 | 18.0% | 61.1% | 0.317 |
| MobileNetV3-Small | 2.3 M | 10.7 | 0.892 | 84.7% | 0.827 | 15.3% | 56.1% | 0.283 |

- Small gains in agreement give large gains in exact-item search: fine-grained ranking
  among near-identical products needs very close embeddings, while tagging doesn't.
- An image–image relational loss made search worse (Recall@10 50.0% vs 56.1%):
  text→image search depends on absolute alignment with the teacher's text space.

## Stage 3: Classification heads

Linear heads on frozen student features, trained on teacher soft labels (T = 4.0),
teacher hard labels, or human labels.

| Training images | Top-1 soft / hard / human | ECE soft / hard |
|---|---|---|
| 1,000 | 83.2 / 84.3 / 88.2% | 0.066 / 0.118 |
| all (33,713) | 84.9 / 84.5 / 93.1% | 0.054 / 0.118 |

- Soft labels halve calibration error vs hard labels and help rare classes when data is
  scarce, but don't raise top-1. Teacher labels plateau at ~3,000 images.
- Human labels (supervised, not distillation) beat the teacher from ~1,000 images,
  partly by matching the test set's own labelling conventions. Mixing 50 human labels
  per class into soft teacher labels gives only 86.3%: the teacher's labels dominate.

## Stage 4: Cascade and deployment

Thresholds chosen on validation; zero-shot student route.

| Deferred to teacher | Top-1 | Macro-F1 | Avg ms / image |
|---|---|---|---|
| 0% (student only) | 85.3% | 0.840 | 20 |
| 1.9% | 85.6% | 0.847 | 43 |
| **9.0%** (threshold 0.656) | **86.0%** | 0.858 | 131 |
| 19.2% | 85.9% | 0.863 | 258 |
| 100% (teacher) | 85.7% | 0.861 | 1,260 |

- The cascade beats the teacher because the two models make partly different mistakes.
- A head trained on soft labels cascaded worse despite better calibration: deferral
  needs confidence that *ranks* errors well, and that head copies the teacher's errors.
- ONNX export: identical predictions (min cosine 0.9999998), 20.9 MB, 2.7 ms vs 12.8 ms
  in PyTorch (model only). Image preprocessing now costs more than the model.
- In-browser demo: the teacher's text tower is exported to ONNX and int8-quantised
  (125 MB). Search Recall@10 drops only from 61.1% to 60.9% (`results/web_search_check.json`).

## Limitations

- Search still needs the teacher's text tower, and deferrals need its image tower.
- Latencies come from one laptop on dataset-sized images. Large web photos add
  preprocessing time, and unfamiliar images are deferred more often than 9%.
- One dataset; the cascade's gain over the teacher is small (0.3 points).
- The web demo tags with the student only (no cascade), searches with the teacher's
  int8 text tower, has a fixed read-only catalog, and downloads ~170 MB on the first visit.

## Future work

- Distill the text tower too, removing the teacher from search.
- Log deferrals and retrain the student on them to lower the deferral rate.
- An attribute-based search benchmark ("grey wallet for women").
- int8 quantisation of the student and the teacher's image tower (the text tower is done).

## Reproduce

```bash
# Install CUDA PyTorch first (pytorch.org), then:
pip install -e ".[dev,app,deploy]"

python scripts/resize_images.py --src <kaggle-images> --dst data/images_256
python scripts/prepare_data.py          # clean, merge classes, split
python scripts/cache_teacher.py         # teacher embeddings + logits (once)
python scripts/eval_teacher.py --split test
python scripts/train_student.py         # settings in configs/base.yaml
python scripts/eval_student.py
python scripts/cache_student_features.py
python scripts/run_head_experiments.py  # Stage 3
python scripts/run_cascade.py           # Stage 4
python scripts/export_onnx.py
python scripts/ingest.py --encoder student
python scripts/run_app.py               # cascade demo
python scripts/build_web.py             # in-browser demo -> build/web
python scripts/build_web.py --push USER/SPACE   # upload it (after `hf auth login`)
```