"""Convert the supplementary simulation GIFs into small, cropped MP4s for the web page.

The GIFs are rendered on a chroma-key green background (exactly RGB 0,255,0).
This script replaces the background with white, crops every scene with one fixed
box (so all methods of a scene share the same framing; the boxes are zoomed in like
the hero clips) and encodes H.264 MP4s plus a poster image. Only test task 2 is used.

Usage:  python tools/convert_sim_gifs.py <videos_of_simulations dir> <docs/static/videos/sim>
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

# folder name in the supplementary material -> method id used on the web page
METHOD_IDS = {
    "PEACH": "peach",
    "PSTNET_Encoder": "pstnet", "PSTNET": "pstnet",
    "GNN_Encoder": "gnn", "GNN": "gnn",
    "MANGO": "mango",
    "Oracle_Mango": "oracle",
    "MGN_Oracle": "oracle_mgn",
    "No_Context": "nocontext", "No_context": "nocontext",
    "MGN": "nocontext_mgn",
}
BG = np.array([0, 255, 0])
TASK = "task_2"
# crop box (x0, y0, x1, y1) in GIF pixels per scene. The trampoline is zoomed in on the sheet,
# so the ball is cut off at the top while it is still high above the sheet.
CROPS = {
    "bending_beam": (12, 0, 606, 222),
    "deforming_block": (202, 0, 517, 336),
    "sheet_deformation": (180, 132, 602, 326),
    "trampoline": (160, 150, 569, 402),
}
TARGET_WIDTH = 640  # output width in px (upscaled by at most 1.5x)


def load_frames(path):
    im = Image.open(path)
    frames, delays = [], []
    for fr in ImageSequence.Iterator(im):
        frames.append(np.asarray(fr.convert("RGB")))
        delays.append(fr.info.get("duration", 100))
    return np.stack(frames), delays


def main(src, dst):
    src, dst = Path(src), Path(dst)
    for scene_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        gifs = sorted(scene_dir.glob(f"*/{TASK}/{TASK}.gif"))
        data = {g: load_frames(g) for g in gifs}
        x0, y0, x1, y1 = CROPS[scene_dir.name]
        delays = next(iter(data.values()))[1]
        fps = 1000.0 / float(np.median(delays))
        print(f"{scene_dir.name}: crop x[{x0},{x1}] y[{y0},{y1}] fps={fps:.2f} gifs={len(gifs)}")

        for g, (frames, _) in data.items():
            method = METHOD_IDS[g.parent.parent.name]
            out_dir = dst / scene_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            crop = frames[:, y0:y1, x0:x1].copy()
            mask = np.all(crop == BG, axis=-1)
            crop[mask] = 255
            ch, cw = crop.shape[1:3]
            scale = min(1.5, TARGET_WIDTH / cw)
            ow, oh = int(cw * scale) // 2 * 2, int(ch * scale) // 2 * 2
            out = out_dir / f"{method}.mp4"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{cw}x{ch}", "-r", f"{fps:.4f}", "-i", "-",
                   "-vf", f"scale={ow}:{oh}:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
                   "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
            subprocess.run(cmd, input=crop.tobytes(), check=True)
            Image.fromarray(crop[0]).resize((ow, oh), Image.LANCZOS).save(
                out.with_suffix(".jpg"), quality=85)


if __name__ == "__main__":
    main(*sys.argv[1:3])
