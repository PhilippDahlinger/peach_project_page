#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Rank-based soft OOD splits.")
    parser.add_argument("hdf5_file")
    parser.add_argument("--property", required=True)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-o", "--overwrite", action="store_true")
    return parser.parse_args()


def load_properties(f, indices, prop_name):
    props = []
    for idx in indices:
        key = f"task_{idx:03d}/params/{prop_name}"
        if key not in f:
            raise ValueError(f"Missing property at {key}")
        props.append(float(f[key][()]))
    return np.array(props)


def compute_rank_probabilities(sorted_indices, alpha):
    n = len(sorted_indices)
    positions = np.arange(n)

    center = n // 2
    dist = np.abs(positions - center)

    if dist.max() > 0:
        d = dist / dist.max()
    else:
        d = dist

    p_train = np.exp(-alpha * d)
    return p_train


def create_splits(indices, props, n_train, n_val, n_test, alpha, rng):
    # --- sort by property ---
    order = np.argsort(props)
    sorted_indices = indices[order]
    sorted_props   = props[order]

    # --- rank-based probabilities ---
    p_train = compute_rank_probabilities(sorted_indices, alpha)

    # normalize for sampling
    weights = p_train / p_train.sum()

    # --- sample train ---
    train = rng.choice(sorted_indices, size=n_train, replace=False, p=weights)

    # --- remaining ---
    mask = np.isin(sorted_indices, train, invert=True)
    remaining = sorted_indices[mask]

    # --- split remaining randomly ---
    rng.shuffle(remaining)

    val = remaining[:n_val]
    test = remaining[n_val:n_val + n_test]

    return train, val, test, sorted_indices, sorted_props



import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def visualize(sorted_props, sorted_indices, train, val, test):
    idx_to_pos = {idx: i for i, idx in enumerate(sorted_indices)}

    train_pos = [idx_to_pos[i] for i in train]
    eval_pos  = [idx_to_pos[i] for i in list(val) + list(test)]

    train_props = sorted_props[train_pos]
    eval_props  = sorted_props[eval_pos]

    fig, ax = plt.subplots(figsize=(6, 2.5))

    ax.scatter(train_props, np.zeros_like(train_props),
               color="#2166ac", s=200, marker="|", linewidths=3,
               label="Train", zorder=2)
    ax.scatter(eval_props, np.zeros_like(eval_props),
               color="#d6604d", s=200, marker="|", linewidths=3,
               label="Eval", zorder=3)

    ax.set_xlabel("Poisson Ratio")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, markerscale=1.5,
              loc="upper left", bbox_to_anchor=(0, 1.6),
              ncol=2)

    fig.tight_layout()
    fig.savefig("split_visualization.pdf", bbox_inches="tight")
    plt.show()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    try:
        f = h5py.File(args.hdf5_file, "r+")
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    with f:
        splits = f["splits"]

        train_old = splits["train_indices"][:]
        val_old   = splits["val_indices"][:]
        test_old  = splits["test_indices"][:]

        n_train = len(train_old)
        n_val   = len(val_old)
        n_test  = len(test_old)

        all_indices = np.concatenate([train_old, val_old, test_old])
        props = load_properties(f, all_indices, args.property)

        indices = all_indices.astype(int)

        train_new, val_new, test_new, sorted_indices, sorted_props = create_splits(
            indices, props, n_train, n_val, n_test, args.alpha, rng
        )

        print(f"Train: {len(train_new)}")
        print(f"Val  : {len(val_new)}")
        print(f"Test : {len(test_new)}")

        # if "ood_splits" in f:
        #     if not args.overwrite:
        #         print("ood_splits exists. Use -o.", file=sys.stderr)
        #         sys.exit(1)
        #     del f["ood_splits"]
        #
        # grp = f.create_group("ood_splits")
        # grp.create_dataset("train_indices", data=train_new)
        # grp.create_dataset("val_indices",   data=val_new)
        # grp.create_dataset("test_indices",  data=test_new)
        #
        # print("Saved 'ood_splits'.")

        visualize(sorted_props, sorted_indices, train_new, val_new, test_new)


if __name__ == "__main__":
    main()