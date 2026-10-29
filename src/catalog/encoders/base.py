from abc import ABC, abstractmethod

import numpy as np
import torch

class Encoder(ABC):
    name: str
    preprocess: callable

    @abstractmethod
    def encode_images(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (embeddings (N, D), L2-normalised; class logits (N, C))."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return (M, D) L2-normalised text embeddings."""