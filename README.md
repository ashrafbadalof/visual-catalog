## Teacher baseline (ceiling)

Teacher: OpenCLIP ViT-L-14 (`laion2b_s32b_b82k`), zero-shot, 51 classes.
Evaluated on the held-out test split (human labels, 6,322 images).

| Metric | Score |
|---|---|
| Top-1 accuracy | 85.7% |
| Top-5 accuracy | 99.7% |
| Macro-F1 | 0.861 |
| ECE (calibration) | 0.051 |
| Text→image Recall@1 / @5 / @10 (exact) | 27.6% / 56.5% / 69.9% |
| Text→image MRR (exact) | 0.413 |
| Text→image Recall@10 (same-title match) | 74.1% |
| Speed (RTX 3050 laptop, batch 32) | 17.6 img/s (57 ms/image) |
| Tagging latency (CPU, single image) | ≈ 1.24 s |
| Search latency (CPU, text tower + database) | ≈ 210 ms |

Prompt names and class merges (Kurtis/Tunics → Kurtas, Flats → Sandals)
were tuned on the validation split only. Full results in `results/`.

## Stage 2: Embedding students

Students are trained to reproduce the teacher's image embeddings (cosine loss on
cached targets, 20 epochs, batch 32). Text embeddings still come from the teacher.
All results on the test split; CPU latency is single-image, preprocessing included.

| Model | Params | CPU ms/img | Agreement | Top-1 | Macro-F1 | ECE | Recall@1 | Recall@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| Teacher (ViT-L-14) | ~300 M | 1,240 | — | 85.7% | 0.861 | 0.051 | 27.6% | 69.9% | 0.413 |
| MobileNetV3-Large | 5.2 M | 19.7 | 0.906 | 85.3% | 0.840 | 0.059 | 18.0% | 61.1% | 0.317 |
| MobileNetV3-Small | 2.3 M | 10.7 | 0.892 | 84.7% | 0.827 | 0.062 | 15.3% | 56.1% | 0.283 |

Recall and MRR are exact-item text→image retrieval. The large student keeps 99.5% of
the teacher's top-1 and 87% of its Recall@10 while running ~63× faster on CPU.

**Other runs (MobileNetV3-Small):**
- 5 epochs instead of 20: agreement 0.870, Recall@10 47.8%. Longer training helped.
- Adding an image–image relational loss: agreement 0.876, Recall@10 50.0%, worse
  than cosine alone. Search is text→image, so it depends on absolute alignment with
  the teacher's text space, which relational losses don't constrain.

## Stage 3: Classification heads

Linear heads on frozen MobileNetV3-Large features, trained on teacher soft labels
(T = 4.0), teacher hard labels, or human labels. Test split.

| Training images | Top-1 soft / hard / human | ECE soft / hard |
|---|---|---|
| 1,000 | 83.2 / 84.3 / 88.2% | 0.066 / 0.118 |
| all (33,713) | 84.9 / 84.5 / 93.1% | 0.054 / 0.118 |
| Teacher | 85.7% | 0.051 |

- Soft labels match hard labels on accuracy but halve calibration error, and improve
  macro-F1 by up to 10 points when data is scarce (100 images).
- Teacher labels plateau at ~3,000 images, below the teacher; heads trained on them
  copy its errors (Heels recall 0.37 vs teacher 0.38).
- Human labels (supervised, not distillation) beat the teacher from ~1,000 images and
  fix its weak classes (Heels recall 0.86), partly by matching the test set's own
  labelling conventions.
- Adding 50 human labels per class to soft teacher labels gives only 86.3%: the
  teacher's labels outweigh the corrections.

Full results: `results/stage3_heads.csv`.