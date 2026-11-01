import open_clip
import torch

from catalog.encoders.base import Encoder

class ClipTeacher(Encoder):
    def __init__(self, arch, pretrained, class_names, prompts, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )

        self.logit_scale = model.logit_scale.exp().item()
        self.model = model.to(self.device, dtype=self.dtype).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.name = f"{arch}/{pretrained}"
        self.class_embs = self._class_embeddings(class_names, prompts)

    @torch.no_grad()
    def _encode_text(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        embs = self.model.encode_text(tokens).float()
        return embs / embs.norm(dim=-1, keepdim=True)

    def _class_embeddings(self, class_names, prompts):
        embs = []
        for name in class_names:
            e = self._encode_text([p.format(name) for p in prompts]).mean(0)
            embs.append(e / e.norm())
        return torch.stack(embs)

    def embed_texts(self, texts, batch_size=256):
        parts = [self._encode_text(texts[i:i + batch_size])
                 for i in range(0, len(texts), batch_size)]
        return torch.cat(parts).cpu().numpy()

    @torch.no_grad()
    def encode_images(self, batch):
        batch = batch.to(self.device, dtype=self.dtype)
        embs = self.model.encode_image(batch).float()
        embs = embs / embs.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * embs @ self.class_embs.T
        return embs.cpu().numpy(), logits.cpu().numpy()
        
