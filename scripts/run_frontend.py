#!/usr/bin/env python3
"""Launcher script for CommonsCat AI Stage 1 & Stage 2 Frontend Studio."""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.web.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Start CommonsCat AI Studio (Stage 1 & Stage 2 Frontend)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Launching CommonsCat AI Studio on http://localhost:{args.port} ...")
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
