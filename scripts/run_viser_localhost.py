"""Launch RADIO-ViPE's viser visualizer bound to 0.0.0.0 so VS Code port
forwarding (which connects to localhost on the remote) can reach it.

Usage:
    python scripts/run_viser_localhost.py <results_dir> [--port 20540]
"""

import argparse
from pathlib import Path

import vipe.utils.viser as viser_mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("vipe_results"))
    parser.add_argument("--port", type=int, default=20540)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    viser_mod.get_host_ip = lambda: args.host
    viser_mod.run_viser(args.results_dir, args.port)


if __name__ == "__main__":
    main()
