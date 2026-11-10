from catalog.config import load_config
from catalog.cache.student_features import run

if __name__ == "__main__":
    run(load_config())