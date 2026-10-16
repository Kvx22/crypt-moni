"""Launcher that works from any folder:  python run.py up   (Windows: py run.py up)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)            # config, data/ and the KILL file are relative to this folder
sys.path.insert(0, HERE)


def load_env(path=".env"):
    """Read KEY=VALUE lines from .env into the environment (real environment variables win)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


load_env()
from cryptomoni.__main__ import main  # noqa: E402

main()
