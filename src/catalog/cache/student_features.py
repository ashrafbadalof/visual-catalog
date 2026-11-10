import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from catalog.config import data_path
from catalog.data.dataset import ImageDataset
from catalog.encoders.student import load_student


@torch.no_grad()
def run(cfg):
    run_name = cfg["head"]["student_run"]
    out = data_path(cfg, "features_dir")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_path(cfg, "processed_dir") / "splits.parquet")

    student = load_student(data_path(cfg, "checkpoints_dir") / f"{run_name}.pt",
                           data_path(cfg, "cache_dir"), device="cuda")
    student.net.classifier[-1] = nn.Identity()

    loader = DataLoader(ImageDataset(df["path"], data_path(cfg, "images_dir"), student.preprocess),
                        batch_size=128, num_workers=4)
    feats = [student.net(batch.to("cuda")).float().cpu().numpy()
             for batch in tqdm(loader, desc="Features")]

    np.save(out / f"{run_name}.npy", np.concatenate(feats).astype(np.float16))
    np.save(out / "ids.npy", df["id"].to_numpy())