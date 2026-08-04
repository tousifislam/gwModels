#!/usr/bin/env python
"""
Verify that all required gwModels data files are present and intact.

Usage:
    python gwmodels_setup.py          # check from repo root
    python gwmodels_setup.py /path/to/data   # check a specific data directory
"""

import os
import sys
import hashlib

# Expected files: {filename: (md5, size_bytes)}
EXPECTED_FILES = {
    "SXSBBH1355.npy": ("222849cb768c5a295a9a52178f5fc6c8", 3142641),
    "gwEccEvolve_NoSpinq4.npy": ("787934f1676c800ab774793e6b55f4c5", 1902416),
    "gwModel_kick_prec_flow.pt": ("fee7e290eba19a2e11cb30750571e05a", 31034),
    "gwModel_kick_prec_flow_config.npy": ("b052c4f12e4bb29ffe5ee3680a791f1e", 592),
    "gwModel_kick_q200_GPR_aligned_spin.pkl": ("564fcc2c2e5eadd7656825b399dced83", 1779153),
    "gwModelRemP_flow.pt": ("82ec1cccb4b5c2e621cc47d0aca95dd4", 1321630),
}


def md5sum(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_data(datadir="data"):
    datadir = os.path.abspath(datadir)
    print(f"Checking data directory: {datadir}\n")

    all_ok = True
    for filename, (expected_md5, expected_size) in sorted(EXPECTED_FILES.items()):
        filepath = os.path.join(datadir, filename)

        if not os.path.isfile(filepath):
            print(f"  MISSING  {filename}")
            all_ok = False
            continue

        actual_size = os.path.getsize(filepath)
        if actual_size != expected_size:
            print(f"  SIZE MISMATCH  {filename}  (expected {expected_size}, got {actual_size})")
            all_ok = False
            continue

        actual_md5 = md5sum(filepath)
        if actual_md5 != expected_md5:
            print(f"  MD5 MISMATCH   {filename}  (expected {expected_md5}, got {actual_md5})")
            all_ok = False
            continue

        print(f"  OK  {filename}  ({actual_size:,} bytes)")

    print()
    if all_ok:
        print(f"All {len(EXPECTED_FILES)} data files verified.")
    else:
        print("Some files are missing or corrupted. See above.")
    return all_ok


if __name__ == "__main__":
    datadir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("gwModels", "data")
    ok = check_data(datadir)
    sys.exit(0 if ok else 1)
