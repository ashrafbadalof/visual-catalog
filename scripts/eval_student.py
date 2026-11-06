from catalog.config import load_config
from catalog.eval.student_eval import run

if __name__ == "__main__":
    run(load_config())