import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from catalog.config import data_path
from catalog.data.dataset import ImageDataset
from catalog.encoders.student import build_student_net, student_preprocess

@torch.no_grad()
def agreement(net, loader, device):
    net.eval()
    sims = []
    for images, target in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(images.to(device))
        sims.append(F.cosine_similarity(out.float(), target.to(device)).cpu())
    return torch.cat(sims).mean().item()

def run(cfg):
    s = cfg["student"]
    cache = data_path(cfg, "cache_dir")
    df = pd.read_parquet(data_path(cfg, "processed_dir") / "splits.parquet")
    assert (np.load(cache / "ids.npy") == df["id"].to_numpy()).all(), "Cache out of sync"

    targets = np.load(cache / "image_embs.npy").astype(np.float32)
    targets = torch.from_numpy(targets / np.linalg.norm(targets, axis=1, keepdims=True))

    def loader(split, transform, shuffle):
        mask = (df["split"] == split).to_numpy()
        ds = ImageDataset(df.loc[mask, "path"], data_path(cfg, "images_dir"),
                          transform, targets[torch.from_numpy(mask)])
        return DataLoader(ds, batch_size=s["batch_size"], shuffle=shuffle,
                          num_workers=s["num_workers"], persistent_workers=True)

    train_dl = loader("train", T.Compose([T.RandomHorizontalFlip(), student_preprocess()]), True)
    val_dl = loader("val", student_preprocess(), False)

    device = "cuda"
    net = build_student_net(s["arch"], targets.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s["lr"],
                                                total_steps=s["epochs"] * len(train_dl))
    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = data_path(cfg, "checkpoints_dir")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{s['run_name']}.pt"
    best = -1.0

    for epoch in range(s["epochs"]):
        net.train()
        total = 0.0
        for images, target in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{s['epochs']}"):
            images, target = images.to(device), target.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(images)
            loss = 1 - F.cosine_similarity(out.float(), target).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()

        val_cos = agreement(net, val_dl, device)
        print(f"Epoch {epoch + 1}: train loss {total / len(train_dl):.4f}, val cosine {val_cos:.4f}")
        if val_cos > best:
            best = val_cos
            torch.save({"arch": s["arch"], "dim": targets.shape[1],
                        "state_dict": net.state_dict(), "val_cosine": val_cos}, ckpt_path)  
    print(f"Best val cosine {best:.4f} → {ckpt_path}")