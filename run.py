"""Launcher that works from any folder:  python run.py up   (Windows: py run.py up)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)            # config, data/ and the KILL file are relative to this folder
sys.path.insert(0, HERE)
from cryptomoni.__main__ import main  # noqa: E402

main()
