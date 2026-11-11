import onnxruntime as ort

from catalog.encoders.base import Encoder
from catalog.encoders.student import student_preprocess

class OnnxStudentEncoder(Encoder):
    def __init__(self, onnx_path, name, text_fn=None):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.preprocess = student_preprocess()
        self.name = name
        self.device = "cpu"
        self.text_fn = text_fn

    def encode_images(self, batch):
        embs, logits = self.session.run(None, {"image": batch.numpy()})
        return embs, logits

    def embed_texts(self, texts):
        return self.text_fn(texts)