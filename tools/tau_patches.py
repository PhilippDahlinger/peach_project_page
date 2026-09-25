"""Precompute the space-time patches for the interactive tau explainer on the project page.

For every slider step (a value of the time scale tau) this script
  1. embeds all points of a small 1D point cloud sequence in space-time (x, h, tau * t),
  2. samples 14 patch centers with farthest point sampling (FPS), always starting from the
     same seed point (middle frame, a quarter along the profile),
  3. assigns every point to its nearest center.

Colors: the first step (smallest tau) gets a coloring where touching patches have clearly
different colors. For every following step, patches are matched to the colors of the previous
step by solving a linear assignment problem on point overlaps, so the number of points that
change color between neighboring slider steps is as small as possible.

The page (docs/static/js/main.js) only reads the result:
    docs/static/data/tau_patches.json

Usage:  python tools/tau_patches.py [--out docs/static/data/tau_patches.json]
Requires numpy and scipy.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

N_FRAMES = 7          # rows of the waterfall, frame 0 at the top
N_PER_FRAME = 34
N_CENTERS = 14        # = 2 per frame, which is what large tau converges to
TAU_MIN, TAU_MAX, N_STEPS = 0.3, 3.0, 121
DIP = 0.14            # depth of the final deformation
SEED = 11

# well-separated categorical colors (picked greedily below for maximal Lab distance)
CANDIDATES = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#393b79", "#e7298a", "#66a61e", "#a6761d", "#1b9e77", "#7570b3",
    "#f0c808", "#003f5c", "#ff9da7", "#9c755f", "#86bcb6", "#b07aa1", "#59a14f", "#edc948",
]


def profile(x, t):
    """A straight 1D profile that is pushed down progressively (it never springs back)."""
    return -DIP * t * np.exp(-((x - 0.5) ** 2) / 0.03)


def make_points(rng):
    xs, ts, fs = [], [], []
    for f in range(N_FRAMES):
        t = f / (N_FRAMES - 1)
        x = np.sort(0.02 + 0.96 * rng.random(N_PER_FRAME))
        xs.append(x)
        ts.append(np.full(N_PER_FRAME, t))
        fs.append(np.full(N_PER_FRAME, f))
    x, t, f = map(np.concatenate, (xs, ts, fs))
    return x, profile(x, t), t, f


def fps(coords, first, k):
    centers = [first]
    d = np.linalg.norm(coords - coords[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        centers.append(nxt)
        d = np.minimum(d, np.linalg.norm(coords - coords[nxt], axis=1))
    return centers


def srgb_to_lab(hex_color):
    rgb = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ lin
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def pick_palette(k):
    labs = {c: srgb_to_lab(c) for c in CANDIDATES}
    chosen = CANDIDATES[:1]
    while len(chosen) < k:
        best = max((c for c in CANDIDATES if c not in chosen),
                   key=lambda c: min(np.linalg.norm(labs[c] - labs[o]) for o in chosen))
        chosen.append(best)
    return chosen


def contact_matrix(coords, labels, k, n_neighbors=4):
    """How strongly two patches touch: count of nearest-neighbor pairs across patches."""
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :n_neighbors]
    w = np.zeros((k, k))
    for i, js in enumerate(nn):
        for j in js:
            if labels[i] != labels[j]:
                w[labels[i], labels[j]] += 1
                w[labels[j], labels[i]] += 1
    return w


def initial_coloring(contact, palette, rng, iters=4000):
    """Color permutation maximizing the Lab distance between touching patches (local search)."""
    labs = np.array([srgb_to_lab(c) for c in palette])
    dist = np.linalg.norm(labs[:, None] - labs[None], axis=-1)
    k = len(palette)

    def score(perm):
        # emphasize the worst contact: a soft minimum over touching pairs
        s = 0.0
        for a, b in zip(*np.nonzero(np.triu(contact))):
            s += contact[a, b] * np.exp(-dist[perm[a], perm[b]] / 25)
        return -s

    best = list(range(k))
    best_s = score(best)
    for _ in range(iters):
        cand = best.copy()
        i, j = rng.choice(k, 2, replace=False)
        cand[i], cand[j] = cand[j], cand[i]
        s = score(cand)
        if s > best_s:
            best, best_s = cand, s
    return best  # patch index -> palette index


def main(out):
    rng = np.random.default_rng(SEED)
    x, h, t, f = make_points(rng)
    # Seed point: middle frame, a quarter along the profile. Seeding in the middle frame makes
    # FPS bisect the frames symmetrically, so for large tau every frame gets exactly one left and
    # one right center. (Seeding in an outer frame puts some first centers mid-profile, between
    # neighbors whose centers sit on opposite ends, which leaves frames with 1 and 3 patches.)
    in_row = np.nonzero(f == N_FRAMES // 2)[0]
    first = int(in_row[np.argmin(np.abs(x[in_row] - 0.25))])

    taus = np.geomspace(TAU_MIN, TAU_MAX, N_STEPS)
    palette = pick_palette(N_CENTERS)
    steps, prev_colors = [], None
    changed_total = 0
    for tau in taus:
        coords = np.stack([x, h, tau * t], axis=1)
        centers = fps(coords, first, N_CENTERS)
        d = np.linalg.norm(coords[:, None] - coords[centers][None], axis=-1)
        labels = np.argmin(d, axis=1)                        # point -> patch (0..13)
        if prev_colors is None:
            perm = initial_coloring(contact_matrix(coords, labels, N_CENTERS), palette, rng)
        else:
            # overlap[p, c]: points of new patch p that had color c in the previous step
            overlap = np.zeros((N_CENTERS, N_CENTERS))
            np.add.at(overlap, (labels, prev_colors), 1)
            rows, cols = linear_sum_assignment(-overlap)
            perm = [0] * N_CENTERS
            for r, c in zip(rows, cols):
                perm[r] = int(c)
        colors = np.array([perm[p] for p in labels])
        if prev_colors is not None:
            changed_total += int((colors != prev_colors).sum())
        frames_per_patch = np.mean([len(set(f[labels == p])) for p in range(N_CENTERS)])
        per_frame = [len(set(labels[f == k])) for k in range(N_FRAMES)]
        steps.append({
            "tau": round(float(tau), 4),
            "centers": [int(c) for c in centers],
            "colors": "".join("0123456789abcdefghijklmnopqrstuvwxyz"[c] for c in colors),
            "framesPerPatch": round(float(frames_per_patch), 2),
            "singleFrame": bool(max(len(set(f[labels == p])) for p in range(N_CENTERS)) == 1),
            "patchesPerFrame": per_frame,
        })
        prev_colors = colors

    data = {
        "about": "Generated by tools/tau_patches.py; do not edit by hand.",
        "nFrames": N_FRAMES, "dip": DIP, "palette": palette,
        "points": {"x": np.round(x, 4).tolist(), "t": np.round(t, 4).tolist(), "f": f.astype(int).tolist()},
        "steps": steps,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, separators=(",", ":")))
    last = steps[-1]
    print(f"wrote {out}: {N_STEPS} steps, tau {TAU_MIN}..{TAU_MAX}, {len(x)} points")
    print(f"  first step frames/patch {steps[0]['framesPerPatch']}, last step patches per frame {last['patchesPerFrame']}")
    print(f"  color changes: {changed_total} point recolorings over {N_STEPS - 1} transitions "
          f"({changed_total / (N_STEPS - 1):.1f} per step)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/static/data/tau_patches.json")
    main(ap.parse_args().out)
