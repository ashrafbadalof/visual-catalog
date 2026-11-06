import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

from catalog.encoders.base import Encoder

def build_student_net(arch, dim):
    net = getattr(torchvision.models, arch)(weights='DEFAULT')
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, dim)
    return net

def student_preprocess(size=224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

class StudentEncoder(Encoder):
    def __init__(self, net, class_embs, logit_scale, name, device="cpu", text_fn=None):
        self.device = device
        self.net = net.to(device).eval()
        self.class_embs = torch.as_tensor(class_embs, dtype=torch.float32, device=device)
        self.logit_scale = logit_scale
        self.name = name
        self.text_fn = text_fn
        self.preprocess = student_preprocess()

    @torch.no_grad()
    def encode_images(self, batch):
        embs = F.normalize(self.net(batch.to(self.device)).float(), dim=-1)
        logits = self.logit_scale * embs @ self.class_embs.T
        return embs.cpu().numpy(), logits.cpu().numpy()

    def embed_texts(self, texts):
        return self.text_fn(texts)

def load_student(checkpoint, cache_dir, device="cpu", text_fn=None):
    ckpt = torch.load(checkpoint, map_location=device)
    net = build_student_net(ckpt["arch"], ckpt["dim"])
    net.load_state_dict(ckpt["state_dict"])
    meta = json.loads((cache_dir / "metadata.json").read_text())
    return StudentEncoder(net, np.load(cache_dir / "class_embs.npy"), meta["logit_scale"],
                          f"student/{ckpt['arch']}", device, text_fn)

