from catalog.config import load_config
from catalog.store.ingest import run

if __name__ == "__main__":
    run(load_config())