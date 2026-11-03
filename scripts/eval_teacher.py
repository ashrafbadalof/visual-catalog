import argparse

from catalog.config import load_config
from catalog.eval.teacher_eval import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    run(load_config(), p.parse_args().split)