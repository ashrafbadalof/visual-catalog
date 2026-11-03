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

Prompt names and class merges (Kurtis/Tunics → Kurtas, Flats → Sandals)
were tuned on the validation split only. Full results in `results/`.