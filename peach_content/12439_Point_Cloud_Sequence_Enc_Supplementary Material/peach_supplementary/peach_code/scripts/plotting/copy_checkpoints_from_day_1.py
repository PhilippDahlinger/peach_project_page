import shutil
from pathlib import Path

src = Path("output/hydra/training/2026-04-25/5701_bbv_gnn_mango_decoder")
dst = Path("output/hydra/training/2026-04-26/5701d2_bbv_gnn_mango_decoder")

src_subdir = next(src.iterdir())
dst_subdir = next(dst.iterdir())

for seed_dir in src_subdir.iterdir():
    src_ckpt = seed_dir / "checkpoints"
    dst_ckpt = dst_subdir / seed_dir.name / "checkpoints"
    # assert that dst_ckpt exists
    assert dst_ckpt.exists(), f"Destination checkpoint directory {dst_ckpt} does not exist"

    for ckpt in src_ckpt.glob("*.ckpt"):
        if ckpt.name == "last.ckpt":
            continue
        print(f"Copying {ckpt} -> {dst_ckpt / ckpt.name}")
        shutil.copy2(ckpt, dst_ckpt / ckpt.name)

print("Done.")
