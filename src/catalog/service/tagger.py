import csv
import time
from pathlib import Path

from catalog.eval.classification import softmax
from catalog.store.sqlite_store import Product


class Tagger:
    def __init__(self, encoder, store, class_names, data_root, log_path):
        self.encoder = encoder
        self.store = store
        self.class_names = class_names
        self.data_root = Path(data_root)
        self.log_path = Path(log_path)

    def tag_and_add(self, image, title=None):
        image = image.convert("RGB")
        start = time.perf_counter()
        embs, logits = self.encoder.encode_images(self.encoder.preprocess(image).unsqueeze(0))
        ms = self._log("tag", start)

        probs = softmax(logits)[0]
        best = int(probs.argmax())
        product_id = self.store.next_id()
        image_path = f"uploads/{product_id}.jpg"
        (self.data_root / "uploads").mkdir(exist_ok=True)
        image.save(self.data_root / image_path)

        product = Product(id=product_id, image_path=image_path, title=title or None,
                          category=self.class_names[best], confidence=float(probs[best]),
                          encoder=self.encoder.name)
        self.store.add([product], embs)
        top3 = {self.class_names[i]: float(probs[i]) for i in probs.argsort()[::-1][:3]}
        return product, top3, ms

    def search(self, text, k=12, category=None):
        start = time.perf_counter()
        results = self.store.search(self.encoder.embed_texts([text])[0], k=k, category=category)
        return results, self._log("query", start)

    def image_file(self, product):
        return str(self.data_root / product.image_path)

    def _log(self, kind, start):
        ms = (time.perf_counter() - start) * 1000
        new_file = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["kind", "encoder", "device", "ms"])
            writer.writerow([kind, self.encoder.name, self.encoder.device, round(ms, 1)])
        return ms