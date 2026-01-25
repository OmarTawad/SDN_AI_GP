
from pathlib import Path
import sys

# Mock src path
ROOT = Path(".").resolve()
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    from dos_detector.config import load_config
    
    # Needs a dummy config file? or use existing?
    config_path = ROOT / "configs" / "config.yaml"
    if config_path.exists():
        cfg = load_config(config_path)
        print(f"Manifest Path: {cfg.paths.manifest_path}")
        print(f"Manifest Exists: {cfg.paths.manifest_path.exists()}")
    else:
        print("Config file not found")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
