"""Build the data files for the interactive 3D real-world viewer on the project page.

The viewer (docs/static/js/scene-viewer.js) loads one *scene* per viewport:

    <name>.json   metadata (frame times, counts, byte offsets)
    <name>.bin    little-endian binary buffers referenced by the JSON

A scene holds a triangle-mesh trajectory (predicted sheet + ball) and a point
cloud trajectory (the ground-truth sensor observation). Point clouds may have a
different number of points in every frame.

Sub-commands
------------
dummy      Write synthetic placeholder scenes (what the page shows right now).

convert    Convert the ParaView output of the real-world evaluator
           (pc_mango/util/trampoline_real_world_eval/real_world_evaluator.py) to a scene:
             * predicted_trajectory.xdmf  merged sheet+ball mesh, point data
                                          "displacement" and "object_id"
             * gth_pc.pvd                 point-cloud time series (.vtu per frame)
           A .pvd series of meshes works for --mesh as well.

list-pvsm  Print the data files referenced by a ParaView state file (.pvsm), to find
           the inputs for `convert`.

Examples
--------
    python tools/realworld_scene.py dummy --out docs/static/data/realworld

    python tools/realworld_scene.py convert \
        --mesh vis/latex-black/sphere-2_traj-0/predicted_trajectory.xdmf \
        --pc   vis/latex-black/sphere-2_traj-0/gth_pc.pvd \
        --fps 30 --label "PEACH (ours)" \
        --out docs/static/data/realworld --name peach

Requires numpy (and meshio for `convert`).
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# scene writer
# ----------------------------------------------------------------------------
def write_scene(out_dir, name, frame_times, mesh_positions, triangles, mesh_object_id,
                point_frames, point_object_ids=None, label="", meta=None):
    """Write <out_dir>/<name>.json + <name>.bin.

    frame_times       (T,) seconds
    mesh_positions    (T, N, 3) float
    triangles         (F, 3) int
    mesh_object_id    (N,) int, 0 = sheet, 1 = ball (used for colouring)
    point_frames      list of T arrays (P_t, 3)
    point_object_ids  optional list of T arrays (P_t,)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_positions = np.asarray(mesh_positions, np.float32)
    T, N, _ = mesh_positions.shape
    assert len(frame_times) == T == len(point_frames)
    triangles = np.asarray(triangles, np.uint32)
    counts = [len(p) for p in point_frames]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int).tolist()
    points = np.concatenate([np.asarray(p, np.float32).reshape(-1, 3) for p in point_frames])
    if point_object_ids is None:
        point_object_ids = [np.zeros(c) for c in counts]
    pids = np.concatenate([np.asarray(p).reshape(-1) for p in point_object_ids]).astype(np.uint8)

    buffers, blobs, cursor = {}, [], 0
    # float32 / uint32 first, uint8 last so every section stays 4-byte aligned
    for key, arr in [("meshPositions", mesh_positions), ("pointPositions", points),
                     ("meshIndex", triangles), ("meshObjectId", np.asarray(mesh_object_id, np.uint8)),
                     ("pointObjectId", pids)]:
        raw = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes()
        buffers[key] = {"offset": cursor, "length": int(arr.size), "dtype": arr.dtype.name}
        blobs.append(raw)
        cursor += len(raw)
        pad = (-cursor) % 4
        if pad:
            blobs.append(b"\0" * pad)
            cursor += pad

    lo = np.minimum(mesh_positions.reshape(-1, 3).min(0), points.min(0))
    hi = np.maximum(mesh_positions.reshape(-1, 3).max(0), points.max(0))
    manifest = {
        "format": "peach-scene", "version": 1, "label": label, "bin": f"{name}.bin",
        "up": [0, 0, 1], "frameTimes": [round(float(t), 5) for t in frame_times],
        "bounds": [lo.round(5).tolist(), hi.round(5).tolist()],
        "mesh": {"numVertices": int(N), "numTriangles": int(len(triangles))},
        "points": {"frameOffsets": offsets},
        "buffers": buffers, "meta": meta or {},
    }
    (out_dir / f"{name}.bin").write_bytes(b"".join(blobs))
    (out_dir / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out_dir / name}.json/.bin  ({T} frames, {N} vertices, "
          f"{len(triangles)} triangles, {points.shape[0]} points, {cursor / 1e6:.2f} MB)")


# ----------------------------------------------------------------------------
# dummy scenes
# ----------------------------------------------------------------------------
L, R_BALL, G = 0.26, 0.025, 9.81
BALL_XY = np.array([0.14, 0.12])


def grid_sheet(n=41):
    xs = np.linspace(0, L, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    verts = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], 1)
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1
            tris += [[a, b, d], [a, d, c]] if (i + j) % 2 else [[a, b, c], [b, d, c]]
    return verts, np.array(tris)


def icosphere(subdiv=2):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
         [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
         [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f)


def ball_bottom(t, depth, h0=0.08, t_contact_len=0.16, v_rebound=0.55):
    """Height of the ball's lowest point; the sheet dip follows it during contact.

    Free fall, one deep contact phase, then damped hops with short, shallow contacts.
    """
    tc = np.sqrt(2 * h0 / G)
    if t < tc:
        return h0 - 0.5 * G * t ** 2
    s, dur, d, v = t - tc, t_contact_len, depth, v_rebound
    while True:
        if s < dur:  # contact: the ball presses into the sheet
            return -d * np.sin(np.pi * s / dur)
        s -= dur
        hop = 2 * v / G
        if s < hop:  # flight
            return v * s - 0.5 * G * s ** 2
        s -= hop
        dur, d, v = dur * 0.7, d * 0.35, v * 0.4


def sheet_z(verts, dip, center, sigma=0.045):
    x, y = verts[:, 0], verts[:, 1]
    r2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
    clamp = np.sin(np.pi * x / L) * np.sin(np.pi * y / L)  # zero displacement at the frame
    return -dip * np.exp(-r2 / (2 * sigma ** 2)) * clamp / max(
        np.sin(np.pi * center[0] / L) * np.sin(np.pi * center[1] / L), 1e-6)


def simulate(frame_times, depth, sheet, ball, ringing=0.0):
    sv, st = sheet
    bv, bt = ball
    frames = []
    for t in frame_times:
        b = ball_bottom(t, depth)
        dip = max(0.0, -b)
        if ringing and b > 0 and t > 0.3:
            dip = -ringing * np.sin(40 * (t - 0.3)) * np.exp(-8 * (t - 0.3))
        s = sv.copy()
        s[:, 2] = sheet_z(sv, dip, BALL_XY)
        c = np.array([*BALL_XY, b + R_BALL])
        frames.append(np.vstack([s, bv * R_BALL + c]))
    tris = np.vstack([st, bt + len(sv)])
    obj = np.r_[np.zeros(len(sv)), np.ones(len(bv))]
    return np.stack(frames), tris, obj


def observe(frame_times, depth, rng, cam=np.array([0.13, -0.35, 0.40])):
    """Sample a noisy, correspondence-free 'sensor' point cloud of the ground truth."""
    pcs, ids = [], []
    for t in frame_times:
        b = ball_bottom(t, depth)
        uv = rng.uniform(0.01, L - 0.01, size=(1000, 2))
        dip = max(0.0, -b)
        z = sheet_z(uv, dip, BALL_XY)
        sheet = np.c_[uv, z]
        # the ball occludes the bottom of the dip, as in the real recordings
        occluded = np.linalg.norm(uv - BALL_XY, axis=1) < R_BALL * 1.1
        sheet = sheet[~occluded]
        c = np.array([*BALL_XY, b + R_BALL])
        d = rng.normal(size=(400, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = c + R_BALL * d
        vis = ((cam - p) * d).sum(1) > 0
        ball = p[vis][:180]
        pts = np.vstack([sheet, ball]) + rng.normal(scale=0.0015, size=(len(sheet) + len(ball), 3))
        pcs.append(pts)
        ids.append(np.r_[np.zeros(len(sheet)), np.ones(len(ball))])
    return pcs, ids


def make_dummy(out):
    rng = np.random.default_rng(0)
    times = np.arange(20) / 30.0
    sheet, ball = grid_sheet(), icosphere(2)
    gt_depth = 0.045
    pcs, ids = observe(times, gt_depth, rng)
    meta = {"placeholder": True}
    # PEACH: close to the observation (slight under-estimate of the dip)
    pos, tris, obj = simulate(times, gt_depth * 0.93, sheet, ball)
    write_scene(out, "peach", times, pos, tris, obj, pcs, ids, label="PEACH (ours)", meta=meta)
    # No context: regresses to the mean and barely deforms, misaligned with the observation
    pos, tris, obj = simulate(times, 0.010, sheet, ball)
    write_scene(out, "nocontext", times, pos, tris, obj, pcs, ids, label="No Context", meta=meta)


# ----------------------------------------------------------------------------
# ParaView / meshio conversion
# ----------------------------------------------------------------------------
def read_pvd(path):
    root = ET.parse(path).getroot()
    entries = [(float(d.get("timestep", i)), Path(path).parent / d.get("file"))
               for i, d in enumerate(root.iter("DataSet"))]
    return sorted(entries, key=lambda e: e[0])


def read_mesh_series(path):
    """Return times, positions (T,N,3), triangles (F,3), object ids (N,)."""
    import meshio
    path = Path(path)
    if path.suffix == ".pvd":
        times, pos = [], []
        for t, f in read_pvd(path):
            m = meshio.read(f)
            times.append(t)
            pos.append(m.points)
            tris = m.get_cells_type("triangle")
            obj = m.point_data.get("object_id", np.zeros(len(m.points)))
        return np.array(times), np.stack(pos), tris, np.asarray(obj).reshape(-1)
    with meshio.xdmf.TimeSeriesReader(str(path)) as reader:
        points, cells = reader.read_points_cells()
        tris = np.concatenate([c.data for c in cells if c.type == "triangle"])
        times, pos, obj = [], [], None
        for k in range(reader.num_steps):
            t, point_data, _ = reader.read_data(k)
            disp = point_data.get("displacement", np.zeros_like(points))
            p = points + disp
            if p.shape[1] == 2:
                p = np.c_[p, np.zeros(len(p))]
            times.append(t)
            pos.append(p)
            if obj is None and "object_id" in point_data:
                obj = np.asarray(point_data["object_id"]).reshape(-1)
    if obj is None:
        obj = np.zeros(len(points))
    return np.array(times, float), np.stack(pos), tris, obj


def read_point_series(path):
    import meshio
    frames, ids = [], []
    for _, f in read_pvd(path):
        m = meshio.read(f)
        frames.append(np.asarray(m.points)[:, :3])
        oid = next((m.point_data[k] for k in ("object_id", "node_types", "color") if k in m.point_data), None)
        ids.append(np.zeros(len(m.points)) if oid is None or np.ndim(oid) != 1 else oid)
    return frames, ids


def convert(args):
    times, pos, tris, obj = read_mesh_series(args.mesh)
    pcs, ids = read_point_series(args.pc)
    T = min(len(pos), len(pcs))
    if len(pos) != len(pcs):
        print(f"warning: {len(pos)} mesh frames vs {len(pcs)} point-cloud frames, using the first {T}")
    frame_times = np.arange(T) / args.fps if args.fps else times[:T]
    # the evaluator writes the ball as object 1 after the sheet (object 0)
    write_scene(args.out, args.name, frame_times, pos[:T], tris, obj, pcs[:T], ids[:T],
                label=args.label, meta={"source_mesh": Path(args.mesh).name, "source_pc": Path(args.pc).name})


def list_pvsm(args):
    text = Path(args.pvsm).read_text(errors="ignore")
    files = sorted(set(re.findall(r'value="([^"]+\.(?:xdmf|pvd|vtu|vtp|vtk|h5|xmf))"', text)))
    print("\n".join(files) if files else "no data files found in the state file")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dummy")
    d.add_argument("--out", default="docs/static/data/realworld")
    c = sub.add_parser("convert")
    c.add_argument("--mesh", required=True, help="predicted_trajectory.xdmf or a .pvd mesh series")
    c.add_argument("--pc", required=True, help="gth_pc.pvd point-cloud series")
    c.add_argument("--out", default="docs/static/data/realworld")
    c.add_argument("--name", required=True, help="output basename, e.g. peach or nocontext")
    c.add_argument("--label", default="")
    c.add_argument("--fps", type=float, default=30.0, help="frame rate for the time axis (0 = use file times)")
    p = sub.add_parser("list-pvsm")
    p.add_argument("pvsm")
    args = ap.parse_args()
    {"dummy": lambda a: make_dummy(a.out), "convert": convert, "list-pvsm": list_pvsm}[args.cmd](args)


if __name__ == "__main__":
    main()
