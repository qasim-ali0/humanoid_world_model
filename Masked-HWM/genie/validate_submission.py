#!/usr/bin/env python3
"""
submission_checker for compression challenge.

Usage:
    python validate_submission.py submission.zip
"""

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile
from typing import List, Tuple
from tqdm import tqdm

import numpy as np


EXPECTED_SHAPE = (3, 32, 32, 500)
CODEBOOK_SIZE = 64_000
EXPECTED_DTYPES = {"indices": np.int32, "values": np.float32}
DEFAULT_NUM_SAMPLES = 450  # .npz files expected


class ValidationError(Exception):
    """Raised when one or more hard errors are detected."""

def _fail(msg: str, problems: List[str]):
    problems.append(msg)
    print(f"[ERROR] {msg}")


def _check_readme(readme_path: str, problems: List[str]):
    """Parse and sanity‑check README.txt.

    * Must exist.
    * Must be JSON (we allow single quotes + trailing commas, which we normalise).
    * Must contain the six required keys.
    """
    if not os.path.isfile(readme_path):
        _fail("README.txt missing", problems)
        return None

    with open(readme_path, "r", encoding="utf‑8") as f:
        txt = f.read().strip()


    relaxed = re.sub(r"'", '"', txt)
    relaxed = re.sub(r",\s*}" , "}", relaxed)  # trailing comma

    try:
        meta = json.loads(relaxed)
    except json.JSONDecodeError as e:
        _fail(f"README.txt is not valid JSON‑like: {e}", problems)
        meta = None

    required = ["method", "team", "authors", "e-mail", "institution", "country"]
    for field in required:
        if meta is None or field not in meta:
            _fail(f"README.txt missing field: {field}", problems)

    return meta


def _check_npz_file(path: str, problems: List[str]):
    """Verify one <ID>.npz obeys all rules."""
    try:
        data = np.load(path)
    except Exception as e:
        _fail(f"Could not read {os.path.basename(path)}: {e}", problems)
        return

    # required arrays
    for key in ("indices", "values"):
        if key not in data:
            _fail(f"{os.path.basename(path)} missing array '{key}'", problems)
            return

    idx, val = data["indices"], data["values"]

    if idx.shape != EXPECTED_SHAPE:
        _fail(f"{os.path.basename(path)}: indices shape {idx.shape} != {EXPECTED_SHAPE}", problems)
    if val.shape != EXPECTED_SHAPE:
        _fail(f"{os.path.basename(path)}: values shape {val.shape} != {EXPECTED_SHAPE}", problems)

    if idx.dtype != EXPECTED_DTYPES["indices"]:
        _fail(f"{os.path.basename(path)}: indices dtype {idx.dtype}, expected {EXPECTED_DTYPES['indices']}", problems)
    if val.dtype != EXPECTED_DTYPES["values"]:
        _fail(f"{os.path.basename(path)}: values dtype {val.dtype}, expected {EXPECTED_DTYPES['values']}", problems)

    if idx.min() < 0 or idx.max() >= CODEBOOK_SIZE:
        _fail(
            f"{os.path.basename(path)}: indices out of range [0, {CODEBOOK_SIZE - 1}] (min={idx.min()}, max={idx.max()})",
            problems,
        )



def validate(zip_path: str, expected_samples: int = DEFAULT_NUM_SAMPLES) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)

    if os.path.basename(zip_path) != "submission.zip":
        print("[WARN] Zip should be named 'submission.zip' — continuing anyway.")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # Warn about nested dirs, but keep going.
        if any(os.path.isdir(os.path.join(tmp, x)) for x in os.listdir(tmp)):
            print("[WARN] Detected sub‑directories; the evaluator expects a flat layout.")

        _check_readme(os.path.join(tmp, "README.txt"), problems)

        npz_files = [f for f in os.listdir(tmp) if f.endswith(".npz")]
        if len(npz_files) != expected_samples:
            _fail(f"Expected {expected_samples} .npz files, found {len(npz_files)}.", problems)

        for fname in tqdm(sorted(npz_files)):
            _check_npz_file(os.path.join(tmp, fname), problems)

    return len(problems) == 0, problems



def main():
    ap = argparse.ArgumentParser(description="Validate competition submission zip file.")
    ap.add_argument("zip", help="Path to submission.zip")
    ap.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                    help=f"Number of test samples (default {DEFAULT_NUM_SAMPLES})")
    args = ap.parse_args()

    ok, probs = validate(args.zip, args.num_samples)
    if ok:
        print("\nAll checks passed. Good luck in the leaderboard!")
        sys.exit(0)

    print("\nFound problems:")
    for p in probs:
        print("  •", p)
    sys.exit(1)


if __name__ == "__main__":
    main()
