import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

def resize_one(job):
    src, dst, size = job
    if dst.exists():
        return None
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            if scale < 1:
                im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            im.save(dst, "JPEG", quality=90)
        return None
    except Exception as e:
        return f"{src.name}: {e}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--size", type=int, default=256)
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(f, dst / (f.stem + ".jpg"), a.size) for f in src.glob("*.jpg")]
    with ProcessPoolExecutor() as ex:
        failures = [r for r in ex.map(resize_one, jobs, chunksize=64) if r]
    print(f"{len(jobs)} images, {len(failures)} failed")
    for f in failures[:20]:
        print("  ", f)

if __name__ == "__main__":
    main()