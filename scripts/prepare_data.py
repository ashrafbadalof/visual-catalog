from catalog.config import load_config
from catalog.data.prepare import run

if __name__ == "__main__":
    run(load_config())