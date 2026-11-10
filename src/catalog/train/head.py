import torch
import torch.nn as nn
import torch.nn.functional as F


def train_head(X, targets, T=1.0, X_h=None, y_h=None, steps=2000, lr=1e-3, batch=256, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(X.shape[1], targets.shape[1]).to(X.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(steps):
        i = torch.randint(len(X), (batch,), device=X.device)
        log_probs = F.log_softmax(head(X[i]) / T, dim=1)
        loss = -(targets[i] * log_probs).sum(dim=1).mean() * T * T

        if X_h is not None:
            j = torch.randint(len(X_h), (batch,), device=X.device)
            loss = loss + F.cross_entropy(head(X_h[j]), y_h[j])

        opt.zero_grad()
        loss.backward()
        opt.step()
    return head