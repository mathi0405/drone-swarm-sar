#!/usr/bin/env python3
"""One-shot Hugging Face Space deploy for the Swarm-SAR demo.

Reads a WRITE token from the HF_TOKEN environment variable (never a CLI arg, so
it is not stored in shell history), creates the Space if needed, and uploads the
three files in spaces/. Run from the repo root:

    # PowerShell
    $env:HF_TOKEN="hf_your_fresh_write_token"
    python scripts/deploy_hf_space.py

    # bash
    HF_TOKEN=hf_your_fresh_write_token python scripts/deploy_hf_space.py

Optional: --user <hf-username> (default: Chand0504), --space <name> (default: swarm-sar).
The token is used only to authenticate this upload and is never written to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", default="Chand0504", help="Hugging Face username")
    ap.add_argument("--space", default="swarm-sar", help="Space name")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN to a WRITE token first "
              "(https://huggingface.co/settings/tokens).", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    spaces_dir = repo_root / "spaces"
    # Static Space: the only free Space type on HF (Docker/Gradio now need PRO).
    # Serves the self-contained client-side replay viewer — no server needed.
    files = ["index.html", "replay_data.json", "README.md"]
    missing = [f for f in files if not (spaces_dir / f).exists()]
    if missing:
        print(f"ERROR: missing spaces/ files: {missing}\n"
              "Run: python scripts/export_replay.py --out spaces/replay_data.json",
              file=sys.stderr)
        return 2

    repo_id = f"{args.user}/{args.space}"
    api = HfApi(token=token)

    print(f"· creating Space {repo_id} (if needed) …")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static",
                    exist_ok=True)

    for f in files:
        print(f"· uploading {f} …")
        api.upload_file(path_or_fileobj=str(spaces_dir / f), path_in_repo=f,
                        repo_id=repo_id, repo_type="space")

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n[OK] deployed -> {url}")
    print("The Space builds automatically (installs the package from GitHub);")
    print("first build takes a few minutes. Watch progress at the URL above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
