#!/usr/bin/env python3
"""
Subsample train_indices from an HDF5 splits file and write the result
into a new group "small_data_splits" inside the same file.

Usage:
    python subsample_splits.py sd_v1.hdf5 -p 20
    python subsample_splits.py sd_v1.hdf5 -p 20 --seed 42
    python subsample_splits.py sd_v1.hdf5 -p 20 -o    # overwrite existing small_data_splits group
"""

import argparse
import sys
import numpy as np
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Subsample train_indices from 'splits' and write into "
            "'small_data_splits' inside the same HDF5 file."
        )
    )
    parser.add_argument("hdf5_file", help="Path to the HDF5 file (e.g. sd_v1.hdf5)")
    parser.add_argument(
        "-p", "--percentage",
        type=float,
        required=True,
        help="Percentage of train_indices to keep (e.g. 20 for 20%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite 'small_data_splits' if it already exists in the file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0 < args.percentage <= 100):
        print(f"Error: --percentage must be in (0, 100], got {args.percentage}", file=sys.stderr)
        sys.exit(1)

    try:
        f = h5py.File(args.hdf5_file, "r+")
    except OSError as e:
        print(f"Error opening '{args.hdf5_file}': {e}", file=sys.stderr)
        sys.exit(1)

    with f:
        # --- validate splits group ---
        if "splits" not in f:
            print("Error: No 'splits' group found in the HDF5 file.", file=sys.stderr)
            sys.exit(1)

        splits = f["splits"]
        required = {"train_indices", "test_indices", "val_indices"}
        missing = required - set(splits.keys())
        if missing:
            print(f"Error: Missing datasets in 'splits': {missing}", file=sys.stderr)
            sys.exit(1)

        # --- guard against existing small_data_splits group ---
        if "small_data_splits" in f:
            if not args.overwrite:
                print(
                    "Error: 'small_data_splits' group already exists in the file. "
                    "Use -o to overwrite.",
                    file=sys.stderr,
                )
                sys.exit(1)
            del f["small_data_splits"]
            print("Deleted existing 'small_data_splits' group.")

        # --- read ---
        train_indices = splits["train_indices"][:]
        test_indices  = splits["test_indices"][:]
        val_indices   = splits["val_indices"][:]

        # --- subsample ---
        rng = np.random.default_rng(args.seed)
        n_total = len(train_indices)
        n_keep  = max(1, int(round(n_total * args.percentage / 100.0)))

        chosen_positions = rng.choice(n_total, size=n_keep, replace=False)
        chosen_positions.sort()
        train_sub = train_indices[chosen_positions]

        print(f"train_indices : {n_total} total  ->  {n_keep} kept ({args.percentage:.1f}%)")
        print(f"test_indices  : {len(test_indices)} (copied as-is)")
        print(f"val_indices   : {len(val_indices)} (copied as-is)")

        # --- write ---
        grp = f.create_group("small_data_splits")
        grp.create_dataset("train_indices", data=train_sub)
        grp.create_dataset("test_indices",  data=test_indices)
        grp.create_dataset("val_indices",   data=val_indices)

    print(f"\nWrote 'small_data_splits' into '{args.hdf5_file}'.")


if __name__ == "__main__":
    main()