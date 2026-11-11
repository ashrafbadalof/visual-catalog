from catalog.config import load_config
from catalog.deploy.onnx_export import run

if __name__ == "__main__":
    run(load_config())