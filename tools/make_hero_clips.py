"""Tightly cropped PEACH clips for the hero strip of the project page.

Usage:  python tools/make_hero_clips.py <videos_of_simulations dir> <docs/static/videos/hero>
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

# scene, task, crop box (x0, y0, x1, y1) in GIF pixels. The trampoline crop zooms in on
# the sheet, so the ball is cut off at the top while it is still high above the sheet.
CLIPS = {
    "deforming_block": ("task_1", (222, 0, 498, 354)),
    "sheet_deformation": ("task_2", (180, 132, 602, 326)),
    "trampoline": ("task_1", (160, 150, 569, 402)),
}
SCALE = 1.5  # mild upscaling keeps the clips crisp at their display size on HiDPI screens


def main(src, dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for scene, (task, (x0, y0, x1, y1)) in CLIPS.items():
        gif = Image.open(Path(src) / scene / "PEACH" / task / f"{task}.gif")
        frames, delays = [], []
        for fr in ImageSequence.Iterator(gif):
            a = np.asarray(fr.convert("RGB"))[y0:y1, x0:x1].copy()
            a[np.all(a == [0, 255, 0], axis=-1)] = 255
            frames.append(a)
            delays.append(fr.info.get("duration", 100))
        fps = 1000.0 / float(np.median(delays))
        h, w = frames[0].shape[:2]
        ow, oh = int(w * SCALE) // 2 * 2, int(h * SCALE) // 2 * 2
        out = dst / f"{scene}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
                        "-vf", f"scale={ow}:{oh}:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
                        "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)],
                       input=np.stack(frames).tobytes(), check=True)
        Image.fromarray(frames[len(frames) // 2]).resize((ow, oh), Image.LANCZOS).save(out.with_suffix(".jpg"), quality=85)
        print(f"{out}: {ow}x{oh} @ {fps:.2f} fps")


if __name__ == "__main__":
    main(*sys.argv[1:3])
