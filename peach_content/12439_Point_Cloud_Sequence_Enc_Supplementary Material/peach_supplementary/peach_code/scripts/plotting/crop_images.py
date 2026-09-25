from PIL import Image
from pathlib import Path

# === CONFIGURATION ===
input_dirs = [
    Path("output/debug/pc_db_task_95_traj_2/animation"),
]

output_root = Path("output/debug/pc_db_task_95_traj_2/animation_cropped")

# Crop rectangle: (left, upper, right, lower)
# Example: x=100, y=50, width=800, height=600
LEFT = 450
TOP = 150
WIDTH = 575
HEIGHT = 600
# =====================

for input_dir in input_dirs:
    output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob("*.png"):
        with Image.open(img_path) as img:
            # Ensure RGBA to preserve transparency
            img = img.convert("RGBA")

            cropped = img.crop((
                LEFT,
                TOP,
                LEFT + WIDTH,
                TOP + HEIGHT
            ))

            cropped.save(output_dir / img_path.name, format="PNG")

print("Cropping completed.")
