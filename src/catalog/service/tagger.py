import csv
import time
from pathlib import Path

from catalog.eval.classification import softmax
from catalog.store.sqlite_store import Product


class Tagger:
    def __init__(self, encoder, store, class_names, data_root, log_path, fallback=None, threshold=None):
        self.encoder = encoder
        self.store = store
        self.class_names = class_names
        self.data_root = Path(data_root)
        self.log_path = Path(log_path)
        self.fallback = fallback
        self.threshold = threshold

    def tag_and_add(self, image, title=None):
        image = image.convert("RGB")
        start = time.perf_counter()
        embs, logits = self.encoder.encode_images(self.encoder.preprocess(image).unsqueeze(0))
        probs = softmax(logits)[0]
        answered_by = self.encoder.name

        if self.fallback is not None and probs.max() < self.threshold:
            _, logits = self.fallback.encode_images(self.fallback.preprocess(image).unsqueeze(0))
            probs = softmax(logits)[0]
            answered_by = self.fallback.name
        ms = self._log("tag", start, answered_by)

        best = int(probs.argmax())
        product_id = self.store.next_id()
        image_path = f"uploads/{product_id}.jpg"
        (self.data_root / "uploads").mkdir(exist_ok=True)
        image.save(self.data_root / image_path)

        product = Product(id=product_id, image_path=image_path, title=title or None,
                          category=self.class_names[best], confidence=float(probs[best]),
                          encoder=answered_by)
        self.store.add([product], embs)
        top3 = {self.class_names[i]: float(probs[i]) for i in probs.argsort()[::-1][:3]}
        return product, top3, ms

    def search(self, text, k=12, category=None):
        start = time.perf_counter()
        results = self.store.search(self.encoder.embed_texts([text])[0], k=k, category=category)
        return results, self._log("query", start)

    def image_file(self, product):
        return str(self.data_root / product.image_path)

    def _log(self, kind, start, who=None):
        ms = (time.perf_counter() - start) * 1000
        new_file = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["kind", "encoder", "device", "ms"])
            writer.writerow([kind, who or self.encoder.name, self.encoder.device, round(ms, 1)])
        return ms