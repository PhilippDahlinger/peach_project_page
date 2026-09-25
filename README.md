# PEACH project page

Source of the project page for **PEACH: Point Cloud Sequence Encoding for Material-conditioned Graph Network Simulators** (NeurIPS 2026, [arXiv:2605.20978](https://arxiv.org/abs/2605.20978)).

The website is a static page in [`docs/`](docs/). It has no build step and loads nothing from third-party servers: three.js and the fonts (Outfit, Inter, JetBrains Mono, SIL OFL) are self-hosted.

## Publishing with GitHub Pages

1. Repository **Settings → Pages**.
2. *Source*: **Deploy from a branch**, branch `main`, folder **`/docs`**.
3. The page will be served at `https://philippdahlinger.github.io/peach_project_page/`.

Local preview (any static server that supports HTTP range requests, needed for seeking in videos):

```bash
npx http-server docs -p 8000     # or: cd docs && python3 -m http.server 8000
```

## Layout

```
docs/
  index.html                     page content
  static/css/style.css           styles
  static/js/main.js              video comparison player, charts (data inline), widgets
  static/js/scene-viewer.js      three.js viewer for the real-world 3D comparison
  static/js/tau-viewer.js        three.js view of the τ explainer (x, deformation, τ·t axes)
  static/js/vendor/              three.js r169 + OrbitControls, CSS2DRenderer (MIT)
  static/images/                 figures exported from the paper, share card, icons
  static/images/steps/           parts of Figure 1 used in the "How PEACH works" cards
  static/fonts/                  self-hosted web fonts
  static/videos/hero/            tightly cropped PEACH clips for the hero strip
  static/videos/sim/<scene>/     <method>.mp4 + .jpg poster (test task 2)
  static/data/realworld/         3D viewer scenes (<name>.json + <name>.bin)
  static/data/tau_patches.json   precomputed patches/colors for the τ explainer
tools/
  convert_sim_gifs.py            supplementary GIFs -> cropped, white-background MP4s
  make_hero_clips.py             zoomed-in PEACH clips for the hero strip
  tau_patches.py                 FPS patches + color matching for the τ explainer (numpy, scipy)
  realworld_scene.py             3D viewer data: placeholder generator + ParaView/XDMF converter
```

## Common updates

**Link previews.** `og:image`/`twitter:image` point to the absolute URL of `static/images/social_card.png`
(social networks need absolute URLs). Update them if the page moves to another domain.

**Camera-ready paper / new arXiv version.** Update the links in the hero section of `docs/index.html` and the BibTeX block.

**Code release.** Replace the disabled `<span class="btn btn-disabled">…Code…</span>` in `docs/index.html` with
`<a class="btn" href="https://github.com/…">…Code</a>`.

**Real-world 3D data.** The viewer currently shows *placeholder* geometry. To show the real results, export
the files written by `pc_mango/util/trampoline_real_world_eval/real_world_evaluator.py` (the ones the ParaView
state loads) for one trial and one method each:

```bash
pip install numpy meshio h5py
# which files does my ParaView state use?
python tools/realworld_scene.py list-pvsm my_state.pvsm

python tools/realworld_scene.py convert \
    --mesh <vis_dir>/predicted_trajectory.xdmf --pc <vis_dir>/gth_pc.pvd \
    --label "PEACH (ours)" --name peach --out docs/static/data/realworld
python tools/realworld_scene.py convert \
    --mesh <vis_dir_nocontext>/predicted_trajectory.xdmf --pc <vis_dir_nocontext>/gth_pc.pvd \
    --label "No Context" --name nocontext --out docs/static/data/realworld
```

`--mesh` expects the merged sheet + ball mesh with the point data `displacement` (and optionally `object_id`),
`--pc` a `.pvd` series of per-frame point clouds. `--fps 30` sets the time axis (use `--fps 0` to keep the file times).
The placeholder banner disappears automatically once the scenes are no longer marked as placeholders.
To regenerate the placeholder: `python tools/realworld_scene.py dummy`.

**τ explainer.** `python tools/tau_patches.py` regenerates `docs/static/data/tau_patches.json`: for each of the
121 slider steps (τ from 0.1 to 8, geometric spacing) it samples 14 patch centers with FPS from a fixed seed point,
assigns points to their nearest center, and matches patch colors between neighboring steps with a linear
assignment on point overlaps so that as few points as possible change color.

**Simulation videos.** `python tools/convert_sim_gifs.py <videos_of_simulations> docs/static/videos/sim`
re-encodes the supplementary GIFs of test task 2 with fixed, zoomed-in crops per scene (`CROPS` in the script).
They are rendered on an exact chroma green, which is replaced by white.

## Credits

Layout inspired by the [MaNGO project page](https://alrhub.github.io/mango/) and the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (CC BY-SA 4.0).
