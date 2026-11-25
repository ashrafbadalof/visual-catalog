"""Build the in-browser demo into build/web, test it, then upload it as a static HF Space.

    python scripts/build_web.py                              # 1. build
    python -m http.server -d build/web 8000                  # 2. test at http://localhost:8000
    python scripts/build_web.py --push ashrafb04/visual-catalog   # 3. upload the built folder
"""
import argparse
from pathlib import Path

from catalog.config import REPO_ROOT, load_config
from catalog.deploy.web_export import build

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO_ROOT / "build" / "web"))
    p.add_argument("--push", metavar="USER/SPACE", help="upload the built folder to this Space")
    args = p.parse_args()

    out = Path(args.out)
    if not args.push:
        build(load_config(), out)
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.push, repo_type="space",
                          commit_message="Deploy in-browser demo")
        print(f"https://huggingface.co/spaces/{args.push}")
