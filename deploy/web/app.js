// Runs the whole catalog in the browser: no server, only static files.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
env.allowLocalModels = true; // read the tokenizer from ./tokenizer/ on this site
env.allowRemoteModels = false;
env.localModelPath = "./";

const DIM = 768, CONTEXT = 77, EOT = 49407, SIZE = 224;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export async function load() {
  const json = (url) => fetch(url).then((r) => r.json());
  const [products, classes, embs, tokenizer, text, student] = await Promise.all([
    json("catalog/products.json"),
    json("catalog/classes.json"),
    fetch("catalog/embeddings.bin").then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)),
    AutoTokenizer.from_pretrained("tokenizer"),
    ort.InferenceSession.create("models/text_int8.onnx"),
    ort.InferenceSession.create("models/student.onnx"),
  ]);
  return { products, classes, embs, tokenizer, text, student };
}

// Text -> CLIP text embedding -> cosine similarity with every product image.
export async function search(app, query, category = "All", k = 12) {
  const ids = app.tokenizer.encode(query).slice(0, CONTEXT); // adds start/end tokens
  ids[ids.length - 1] = EOT; // same truncation as open_clip
  const tokens = new BigInt64Array(CONTEXT); // zero-padded, like open_clip
  ids.forEach((t, i) => (tokens[i] = BigInt(t)));
  const out = await app.text.run({ tokens: new ort.Tensor("int64", tokens, [1, CONTEXT]) });
  const q = out.embedding.data;

  const hits = [];
  app.products.forEach((p, i) => {
    if (category !== "All" && p.category !== category) return;
    let score = 0;
    for (let d = 0, o = i * DIM; d < DIM; d++) score += q[d] * app.embs[o + d];
    hits.push({ ...p, score });
  });
  return hits.sort((a, b) => b.score - a.score).slice(0, k);
}

// Image -> student (resize, centre crop, ImageNet normalise, as in training) -> top-3 classes.
export async function tag(app, img) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  const scale = SIZE / Math.min(img.naturalWidth, img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  ctx.drawImage(img, (SIZE - w) / 2, (SIZE - h) / 2, w, h);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;

  const x = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++)
    for (let c = 0; c < 3; c++) x[c * SIZE * SIZE + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  const out = await app.student.run({ image: new ort.Tensor("float32", x, [1, 3, SIZE, SIZE]) });

  const logits = Array.from(out.logits.data);
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map((e, i) => ({ name: app.classes[i], prob: e / sum }))
    .sort((a, b) => b.prob - a.prob).slice(0, 3);
}
