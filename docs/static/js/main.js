/* PEACH project page: interactive widgets (no dependencies). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ data
  const METHODS = {
    peach:         { name: "PEACH (ours)",       color: "#1f77b4", group: "pc" },
    pstnet:        { name: "PSTNet Encoder",     color: "#e377c2", group: "pc" },
    gnn:           { name: "GNN Encoder",        color: "#8c564b", group: "pc" },
    tto:           { name: "Test-Time Opt.",     color: "#bcbd22", group: "pc" },
    mango:         { name: "MaNGO",              color: "#17becf", group: "mesh" },
    oracle:        { name: "Oracle",             color: "#2ca02c", group: "oracle" },
    oracle_mgn:    { name: "Oracle (MGN)",       color: "#9467bd", group: "oracle" },
    nocontext:     { name: "No Context",         color: "#7f7f7f", group: "none" },
    nocontext_mgn: { name: "No Context (MGN)",   color: "#ff7f0e", group: "none" },
    peach_noaug:   { name: "PEACH (no data augm.)", color: "#9467bd", group: "pc" },
  };
  const GROUPS = [
    { id: "pc", name: "Point cloud context" },
    { id: "mesh", name: "Mesh context" },
    { id: "oracle", name: "Oracle context" },
    { id: "none", name: "No context" },
  ];
  const SCENES = [
    { id: "deforming_block", name: "Deforming Block", fps: 7.41, frames: 52, ar: "472/504" },
    { id: "sheet_deformation", name: "Sheet Deformation", fps: 7.14, frames: 51, ar: "632/290" },
    { id: "bending_beam", name: "Bending Beam", fps: 14.29, frames: 101, ar: "640/238" },
    { id: "trampoline", name: "Trampoline", fps: 3.57, frames: 25, ar: "612/378" },
  ];
  // methods with a rollout video per scene (supplementary material)
  const VIDEO_METHODS = ["peach", "pstnet", "gnn", "mango", "oracle", "oracle_mgn", "nocontext", "nocontext_mgn"];
  const MISSING = { bending_beam: ["pstnet"], trampoline: ["mango"] };

  // Full rollout MSE, 8 context trajectories: [mean, lower, upper] (paper Fig. 4)
  const MSE = {
    deforming_block: {
      peach: [1.4707e-6, 1.3136e-6, 1.6826e-6], pstnet: [8.7298e-6, 5.6903e-6, 1.1975e-5],
      gnn: [1.9537e-5, 4.8388e-6, 4.4339e-5], tto: [1.2119e-5, 1.0742e-5, 1.3495e-5],
      mango: [1.2112e-6, 1.1531e-6, 1.2742e-6], oracle: [1.2985e-6, 1.1952e-6, 1.4497e-6],
      oracle_mgn: [4.1116e-6, 3.7078e-6, 4.5653e-6], nocontext: [5.0834e-5, 5.0709e-5, 5.0959e-5],
      nocontext_mgn: [5.7853e-5, 5.6436e-5, 5.9399e-5],
    },
    sheet_deformation: {
      peach: [1.45e-7, 1.22e-7, 1.70e-7], pstnet: [2.65e-5, 2.64e-5, 2.66e-5],
      gnn: [1.49e-6, 6.23e-7, 2.43e-6], tto: [4.70e-8, 4.49e-8, 5.00e-8],
      mango: [3.89e-7, 3.34e-7, 4.44e-7], oracle: [4.52e-8, 4.23e-8, 4.81e-8],
      oracle_mgn: [7.92e-7, 6.98e-7, 9.29e-7], nocontext: [2.66e-5, 2.64e-5, 2.69e-5],
      nocontext_mgn: [2.90e-5, 2.85e-5, 2.94e-5],
    },
    bending_beam: {
      peach: [2.68e-4, 2.61e-4, 2.74e-4], pstnet: [1.20e-3, 9.16e-4, 1.51e-3],
      gnn: [8.70e-4, 6.82e-4, 9.86e-4], tto: [5.52e-5, 5.13e-5, 5.90e-5],
      mango: [3.64e-4, 3.31e-4, 4.03e-4], oracle: [2.93e-4, 2.80e-4, 3.07e-4],
      oracle_mgn: [1.85e-3, 1.55e-3, 2.12e-3], nocontext: [7.89e-4, 7.70e-4, 8.12e-4],
      nocontext_mgn: [2.04e-3, 1.70e-3, 2.33e-3],
    },
    trampoline: {
      peach: [1.1675e-5, 1.0020e-5, 1.2944e-5], pstnet: [5.3091e-5, 4.4918e-5, 6.4341e-5],
      gnn: [3.1270e-4, 2.8242e-4, 3.3119e-4], tto: [1.2343e-5, 1.1000e-5, 1.4248e-5],
      mango: [2.0718e-5, 1.7016e-5, 2.4613e-5], oracle: [4.3457e-6, 4.1519e-6, 4.4844e-6],
      oracle_mgn: [1.4253e-4, 1.3208e-4, 1.5335e-4], nocontext: [2.9218e-4, 2.8797e-4, 2.9636e-4],
      nocontext_mgn: [8.5081e-4, 7.3378e-4, 9.8727e-4],
    },
  };
  const MSE_ORDER = ["peach", "pstnet", "gnn", "tto", "mango", "oracle", "oracle_mgn", "nocontext", "nocontext_mgn"];

  // Real world: mean point-to-mesh distance [x 1e-3] at t = 0, 0.04, ..., 0.40 s (paper Fig. 5)
  const RW = {
    peach: { mean: [0.00517, 0.01245, 0.03685, 0.16925, 0.46502, 0.47792, 0.58516, 0.35339, 0.3073, 0.37552, 0.51994], lo: [0.00503, 0.01231, 0.03646, 0.16702, 0.45493, 0.45458, 0.52323, 0.33003, 0.2942, 0.34946, 0.49163], hi: [0.00533, 0.01264, 0.03719, 0.17148, 0.47511, 0.49747, 0.64712, 0.38011, 0.32722, 0.4011, 0.54824] },
    peach_noaug: { mean: [0.00536, 0.01261, 0.03682, 0.16911, 0.46456, 0.50467, 0.70864, 0.41338, 0.30866, 0.3643, 0.50754], lo: [0.00502, 0.01228, 0.03658, 0.16706, 0.44092, 0.48404, 0.66896, 0.40547, 0.30254, 0.36018, 0.49419], hi: [0.00572, 0.01299, 0.03725, 0.17039, 0.47987, 0.52529, 0.73602, 0.42277, 0.31454, 0.36841, 0.5209] },
    oracle: { mean: [0.00502, 0.01227, 0.03663, 0.17186, 0.53636, 0.86493, 0.75647, 0.47337, 0.47714, 0.51604, 0.58717], lo: [0.00501, 0.01224, 0.0365, 0.17015, 0.53206, 0.85425, 0.73765, 0.45989, 0.4716, 0.51268, 0.58334], hi: [0.00503, 0.0123, 0.03672, 0.17357, 0.54405, 0.87941, 0.77891, 0.48684, 0.4814, 0.51941, 0.59102] },
    tto: { mean: [0.00636, 0.01474, 0.03906, 0.18832, 0.52431, 0.97206, 0.97042, 0.52407, 0.42635, 0.4342, 0.5396], lo: [0.00553, 0.01284, 0.03317, 0.12761, 0.44633, 0.83412, 0.85703, 0.37808, 0.37554, 0.40252, 0.4966], hi: [0.00756, 0.01771, 0.04272, 0.24406, 0.59393, 1.11, 1.04278, 0.72817, 0.47715, 0.47157, 0.5826] },
    nocontext: { mean: [0.00617, 0.01336, 0.03793, 0.17288, 0.55071, 1.16768, 1.16615, 0.49832, 0.3802, 0.44616, 0.56058], lo: [0.00512, 0.01245, 0.03704, 0.17009, 0.53873, 1.11711, 1.154, 0.49394, 0.36324, 0.43007, 0.54187], hi: [0.00812, 0.01489, 0.03947, 0.17557, 0.56269, 1.20231, 1.17717, 0.50269, 0.39715, 0.46211, 0.57937] },
    nocontext_mgn: { mean: [0.00526, 0.01322, 0.04098, 0.16844, 0.52522, 1.26656, 1.38343, 0.76764, 0.78091, 0.96148, 1.0907], lo: [0.00513, 0.01269, 0.0388, 0.16726, 0.48638, 1.01408, 1.32488, 0.61874, 0.59531, 0.6982, 0.74745], hi: [0.00541, 0.01375, 0.04352, 0.16992, 0.57129, 1.5753, 1.48175, 1.02241, 1.07759, 1.38303, 1.61138] },
  };
  const RW_NAMES = { oracle: "“Oracle”", tto: "Test-Time Opt." };

  // ------------------------------------------------------------------ helpers
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, children = []) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const c of [].concat(children)) if (c) n.append(c);
    return n;
  };
  const SVGNS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const sci = (v, digits = 2) => {
    const e = Math.floor(Math.log10(v));
    const m = v / 10 ** e;
    return `${m.toFixed(digits)}×10${String(e).split("").map((c) => SUP[c]).join("")}`;
  };
  const pow10 = (e) => `10${String(e).split("").map((c) => SUP[c]).join("")}`;

  function segmented(container, items, selected, onChange) {
    container.innerHTML = "";
    const buttons = items.map((it) => {
      const b = el("button", { role: "tab", "aria-selected": String(it.id === selected), text: it.name, type: "button" });
      b.addEventListener("click", () => {
        buttons.forEach((x) => x.setAttribute("aria-selected", String(x === b)));
        onChange(it.id);
      });
      container.append(b);
      return b;
    });
  }

  function makeTooltip(host) {
    const tt = el("div", { class: "tooltip", role: "status" });
    host.append(tt);
    return {
      show(html, x, y) {
        tt.innerHTML = html;
        tt.classList.add("show");
        const hw = host.clientWidth, tw = tt.offsetWidth, th = tt.offsetHeight;
        let left = x + 14;
        if (left + tw > hw) left = x - tw - 14;
        tt.style.left = `${Math.max(0, left)}px`;
        tt.style.top = `${Math.max(0, y - th / 2)}px`;
      },
      hide() { tt.classList.remove("show"); },
    };
  }

  function onResize(node, cb) {
    let w = node.clientWidth, raf = 0;
    if (!("ResizeObserver" in window)) return;
    new ResizeObserver(() => {
      if (Math.abs(node.clientWidth - w) < 8) return;
      w = node.clientWidth;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(cb);
    }).observe(node);
  }

  function whenVisible(node, cb, opts = { rootMargin: "200px" }) {
    if (!("IntersectionObserver" in window)) { cb(true); return; }
    new IntersectionObserver((entries) => entries.forEach((e) => cb(e.isIntersecting)), opts).observe(node);
  }

  // colored fill for webkit range tracks
  function syncRangeFill(input) {
    const f = () => {
      const p = ((input.value - input.min) / (input.max - input.min)) * 100;
      input.style.setProperty("--fill", `${p}%`);
    };
    input.addEventListener("input", f);
    f();
    return f;
  }
  window.PEACH_syncRangeFill = syncRangeFill;

  // ------------------------------------------------------------------ hero reel: play only when visible
  function initReel() {
    const reel = $(".reel");
    if (!reel) return;
    const vids = [...reel.querySelectorAll("video")];
    whenVisible(reel, (vis) => vids.forEach((v) => (vis ? v.play().catch(() => {}) : v.pause())), { threshold: 0.1 });
  }

  // ------------------------------------------------------------------ synchronized comparison player
  function initPlayer() {
    const root = $("#sim-player");
    if (!root) return;
    const grid = $("#video-grid"), chips = $("#method-chips"), playBtn = $("#play-btn");
    const scrub = $("#scrub"), frameLabel = $("#frame-label");
    const fillScrub = syncRangeFill(scrub);
    const state = { scene: "sheet_deformation", selected: new Set(["peach", "pstnet", "oracle", "nocontext"]), playing: true, rate: 1, visible: false, scrubbing: false };
    let videos = [];

    segmented($("#scene-tabs"), SCENES, state.scene, (id) => { state.scene = id; renderChips(); renderVideos(); });
    segmented($("#speed-tabs"), [{ id: 0.5, name: "0.5×" }, { id: 1, name: "1×" }, { id: 2, name: "2×" }], 1, (r) => { state.rate = r; videos.forEach((v) => (v.playbackRate = r)); });

    function available(m) { return !(MISSING[state.scene] || []).includes(m); }

    function renderChips() {
      chips.innerHTML = "";
      for (const g of GROUPS) {
        const ms = VIDEO_METHODS.filter((m) => METHODS[m].group === g.id);
        if (!ms.length) continue;
        const wrap = el("div", { class: "chip-group" }, el("span", { class: "chip-group-label", text: g.name }));
        for (const m of ms) {
          const ok = available(m);
          const b = el("button", { class: "chip", type: "button", "aria-pressed": String(state.selected.has(m) && ok), title: ok ? "" : "No video for this scene" },
            [el("span", { class: "swatch", style: `--c:${METHODS[m].color}` }), METHODS[m].name]);
          if (!ok) b.disabled = true;
          b.addEventListener("click", () => {
            if (state.selected.has(m)) { if (state.selected.size > 1) state.selected.delete(m); }
            else state.selected.add(m);
            b.setAttribute("aria-pressed", String(state.selected.has(m)));
            renderVideos();
          });
          wrap.append(b);
        }
        chips.append(wrap);
      }
    }

    function renderVideos() {
      const scene = SCENES.find((s) => s.id === state.scene);
      const t0 = videos[0] ? videos[0].currentTime : 0;
      grid.innerHTML = "";
      const ms = VIDEO_METHODS.filter((m) => state.selected.has(m) && available(m));
      grid.className = "video-grid" + (ms.length === 1 ? " n1" : ms.length === 2 ? " n2" : "");
      grid.style.setProperty("--ar", scene.ar);
      videos = ms.map((m) => {
        const base = `static/videos/sim/${scene.id}/${m}`;
        const v = el("video", { src: `${base}.mp4`, poster: `${base}.jpg`, muted: "", playsinline: "", preload: "auto", "aria-label": `${METHODS[m].name} prediction` });
        v.muted = true;
        v.playbackRate = state.rate;
        grid.append(el("div", { class: "vcell" + (m === "peach" ? " is-ours" : "") }, [
          v,
          el("div", { class: "vlabel" }, [el("span", { class: "swatch", style: `--c:${METHODS[m].color}` }), METHODS[m].name]),
        ]));
        return v;
      });
      const master = videos[0];
      master.addEventListener("loadedmetadata", () => {
        const t = Math.min(t0, master.duration || 0);
        videos.forEach((v) => { v.currentTime = t; });
        if (state.playing && state.visible) playAll();
      }, { once: true });
      master.addEventListener("ended", () => {
        videos.forEach((v) => { v.currentTime = 0; });
        if (state.playing) playAll();
      });
    }

    function playAll() { videos.forEach((v) => v.play().catch(() => {})); }
    function pauseAll() { videos.forEach((v) => v.pause()); }
    function setPlaying(p) {
      state.playing = p;
      playBtn.classList.toggle("paused", !p);
      playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
      if (p && state.visible) playAll(); else pauseAll();
    }
    playBtn.addEventListener("click", () => setPlaying(!state.playing));

    scrub.addEventListener("input", () => {
      state.scrubbing = true;
      pauseAll();
      const master = videos[0];
      if (!master || !master.duration) return;
      const t = (scrub.value / 1000) * master.duration;
      videos.forEach((v) => { v.currentTime = Math.min(t, (v.duration || t) - 0.001); });
      updateLabel(t);
    });
    scrub.addEventListener("change", () => { state.scrubbing = false; if (state.playing && state.visible) playAll(); });

    function updateLabel(t) {
      const scene = SCENES.find((s) => s.id === state.scene);
      const f = Math.min(scene.frames, Math.floor(t * scene.fps) + 1);
      frameLabel.textContent = `frame ${String(f).padStart(String(scene.frames).length, " ")} / ${scene.frames}`;
    }

    (function tick() {
      const master = videos[0];
      if (master && master.duration && !state.scrubbing) {
        const t = master.currentTime;
        scrub.value = Math.round((t / master.duration) * 1000);
        fillScrub();
        updateLabel(t);
        // keep followers locked to the master (also while paused, e.g. videos still loading during a scrub)
        for (let i = 1; i < videos.length; i++) {
          const v = videos[i];
          if (v.readyState >= 1 && !v.seeking && Math.abs(v.currentTime - t) > (master.paused ? 0.02 : 0.08)) v.currentTime = t;
          if (!master.paused && v.paused && !v.ended) v.play().catch(() => {});
        }
      }
      requestAnimationFrame(tick);
    })();

    whenVisible(root, (vis) => { state.visible = vis; if (vis && state.playing) playAll(); else pauseAll(); }, { threshold: 0.05 });
    renderChips();
    renderVideos();
  }

  // ------------------------------------------------------------------ MSE bar chart (horizontal, log scale)
  function initMseChart() {
    const host = $("#mse-plot");
    if (!host) return;
    const tt = makeTooltip(host);
    let scene = "deforming_block";
    segmented($("#mse-tabs"), SCENES, scene, (id) => { scene = id; draw(); });

    const tableBtn = $("#mse-table-toggle"), tableWrap = $("#mse-table");
    tableBtn.addEventListener("click", () => {
      const open = tableWrap.hidden;
      tableWrap.hidden = !open;
      tableBtn.setAttribute("aria-expanded", String(open));
      tableBtn.textContent = open ? "Hide table" : "Show table";
    });
    // data table view (all scenes)
    const tbl = el("table", { class: "data" });
    tbl.append(el("thead", {}, el("tr", {}, [el("th", { text: "Method" }), ...SCENES.map((s) => el("th", { text: s.name }))])));
    const tb = el("tbody");
    for (const m of MSE_ORDER) tb.append(el("tr", {}, [el("td", { text: METHODS[m].name }), ...SCENES.map((s) => el("td", { class: "num", text: sci(MSE[s.id][m][0]) }))]));
    tbl.append(tb);
    tableWrap.append(tbl);

    function draw() {
      const data = MSE[scene];
      const W = Math.max(340, Math.min(host.clientWidth || 820, 900));
      const labelW = W < 560 ? 132 : 190, right = W < 560 ? 78 : 96, rowH = 30, groupGap = 26, top = 26;
      const rows = [];
      let y = top;
      for (const g of GROUPS) {
        const ms = MSE_ORDER.filter((m) => METHODS[m].group === g.id);
        rows.push({ type: "group", name: g.name, y: y + 4 });
        y += 20;
        for (const m of ms) { rows.push({ type: "bar", m, y }); y += rowH; }
        y += groupGap - 14;
      }
      const H = y + 30;
      const all = MSE_ORDER.flatMap((m) => [data[m][1], data[m][2]]);
      const e0 = Math.floor(Math.log10(Math.min(...all))), e1 = Math.ceil(Math.log10(Math.max(...all)));
      const x = (v) => labelW + ((Math.log10(v) - e0) / (e1 - e0)) * (W - labelW - right);
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Full rollout MSE per method on ${SCENES.find((q) => q.id === scene).name}` });

      const gridG = svg("g", { class: "grid" });
      for (let e = e0; e <= e1; e++) {
        gridG.append(svg("line", { x1: x(10 ** e), x2: x(10 ** e), y1: top - 6, y2: H - 26 }));
        const t = svg("text", { x: x(10 ** e), y: H - 8, "text-anchor": "middle", class: "tick-label" });
        t.textContent = pow10(e);
        gridG.append(t);
      }
      s.append(gridG);

      const peach = data.peach[0];
      for (const r of rows) {
        if (r.type === "group") {
          const t = svg("text", { x: 0, y: r.y + 8, class: "group-label" });
          t.textContent = r.name.toUpperCase();
          s.append(t);
          continue;
        }
        const [mean, lo, hi] = data[r.m];
        const g = svg("g", { class: "row", tabindex: "0" });
        const cy = r.y + rowH / 2 - 2;
        g.append(svg("rect", { class: "hit", x: 0, y: r.y - 2, width: W, height: rowH, rx: 6 }));
        const lab = svg("text", { x: W < 560 ? 4 : 12, y: cy + 4.5, class: "row-label", "font-weight": r.m === "peach" ? 700 : 400 });
        lab.textContent = METHODS[r.m].name;
        g.append(lab);
        const bx0 = x(10 ** e0), bx1 = x(mean), bh = 16;
        g.append(svg("path", { class: "bar", fill: METHODS[r.m].color, d: roundedRight(bx0, cy - bh / 2, Math.max(bx1 - bx0, 2), bh, 4) }));
        g.append(svg("line", { x1: x(lo), x2: x(hi), y1: cy, y2: cy, stroke: "#2b2b2b", "stroke-width": 1.4 }));
        for (const v of [lo, hi]) g.append(svg("line", { x1: x(v), x2: x(v), y1: cy - 5, y2: cy + 5, stroke: "#2b2b2b", "stroke-width": 1.4 }));
        const vl = svg("text", { x: x(Math.max(hi, mean)) + 8, y: cy + 4, class: "val-label" });
        vl.textContent = sci(mean);
        g.append(vl);
        const ratio = mean / peach;
        const rel = r.m === "peach" ? "reference" : ratio >= 1 ? `${ratio.toFixed(1)}× PEACH's error` : `${(1 / ratio).toFixed(1)}× lower than PEACH`;
        const html = `<div class="tt-title">${METHODS[r.m].name}</div><div class="tt-row">MSE<span class="v">${sci(mean)}</span></div><div class="tt-row">interval<span class="v">${sci(lo, 1)} – ${sci(hi, 1)}</span></div><div class="tt-row">vs. PEACH<span class="v">${rel}</span></div>`;
        const show = (ev) => {
          const b = host.getBoundingClientRect();
          const px = ev && ev.clientX ? ev.clientX - b.left : (bx1 / W) * host.clientWidth;
          const py = ev && ev.clientY ? ev.clientY - b.top : (cy / H) * host.clientHeight;
          tt.show(html, px, py);
        };
        g.addEventListener("mousemove", show);
        g.addEventListener("focus", () => show());
        g.addEventListener("mouseleave", () => tt.hide());
        g.addEventListener("blur", () => tt.hide());
        s.append(g);
      }
      host.querySelector("svg")?.remove();
      host.prepend(s);
    }

    draw();
    onResize(host, draw);
  }
  function roundedRight(x, y, w, h, r) {
    r = Math.min(r, w, h / 2);
    return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
  }

  // ------------------------------------------------------------------ real-world line chart
  function initRwChart() {
    const host = $("#rw-plot");
    if (!host) return;
    const order = ["peach", "peach_noaug", "oracle", "tto", "nocontext", "nocontext_mgn"];
    const visible = new Set(order);
    const tt = makeTooltip(host);
    const legend = $("#rw-legend");
    const nm = (m) => RW_NAMES[m] || METHODS[m].name;
    for (const m of order) {
      const b = el("button", { class: "chip", type: "button", "aria-pressed": "true" }, [el("span", { class: "swatch", style: `--c:${METHODS[m].color}` }), nm(m)]);
      b.addEventListener("click", () => {
        if (visible.has(m)) { if (visible.size > 1) visible.delete(m); } else visible.add(m);
        b.setAttribute("aria-pressed", String(visible.has(m)));
        draw();
      });
      legend.append(b);
    }
    const L = 58, R = 20, T = 16, B = 46;
    const N = 11, ymax = 1.7;
    let W = 820, H = 380;
    const x = (i) => L + (i / (N - 1)) * (W - L - R);
    const y = (v) => T + (1 - v / ymax) * (H - T - B);

    function draw() {
      W = Math.max(340, Math.min(host.clientWidth || 820, 900));
      H = Math.round(Math.max(300, Math.min(400, W * 0.46)));
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Point-to-mesh distance over time on real-world trampoline recordings" });
      // contact phase
      s.append(svg("rect", { x: x(3), y: T, width: x(7) - x(3), height: H - T - B, fill: "#fdecea" }));
      const cp = svg("text", { x: (x(3) + x(7)) / 2, y: T + 16, "text-anchor": "middle", class: "axis-title", fill: "#c0392b", style: "fill:#b83227;font-weight:600" });
      cp.textContent = "ball–sheet contact";
      s.append(cp);
      const g = svg("g", { class: "grid" });
      for (let v = 0; v <= 1.5001; v += 0.25) {
        g.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v) }));
        const t = svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", class: "tick-label" });
        t.textContent = v.toFixed(2);
        g.append(t);
      }
      for (let i = 0; i < N; i += 2) {
        const t = svg("text", { x: x(i), y: H - B + 20, "text-anchor": "middle", class: "tick-label" });
        t.textContent = (i * 0.04).toFixed(2);
        g.append(t);
      }
      s.append(g);
      const xt = svg("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" });
      xt.textContent = "time [s]";
      const yt = svg("text", { x: 14, y: (T + H - B) / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 14 ${(T + H - B) / 2})` });
      yt.textContent = "point-to-mesh distance [×10⁻³]";
      s.append(xt, yt);

      const draws = [...order].reverse().filter((m) => visible.has(m)); // PEACH on top
      for (const m of draws) {
        const d = RW[m];
        const band = d.hi.map((v, i) => `${x(i)},${y(v)}`).concat(d.lo.map((v, i) => `${x(i)},${y(v)}`).reverse());
        s.append(svg("polygon", { points: band.join(" "), fill: METHODS[m].color, opacity: 0.14 }));
      }
      for (const m of draws) {
        const d = RW[m];
        s.append(svg("polyline", { points: d.mean.map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: METHODS[m].color, "stroke-width": m === "peach" ? 3 : 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      }
      const cross = svg("line", { y1: T, y2: H - B, stroke: "#9a8f86", "stroke-dasharray": "3 3", opacity: 0 });
      const dots = svg("g");
      s.append(cross, dots);
      const overlay = svg("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" });
      s.append(overlay);
      const move = (ev) => {
        const b = s.getBoundingClientRect();
        const sx = ((ev.clientX - b.left) / b.width) * W;
        const i = Math.max(0, Math.min(N - 1, Math.round(((sx - L) / (W - L - R)) * (N - 1))));
        cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 1);
        dots.innerHTML = "";
        const ms = order.filter((m) => visible.has(m)).sort((a, c) => RW[a].mean[i] - RW[c].mean[i]);
        for (const m of ms) dots.append(svg("circle", { cx: x(i), cy: y(RW[m].mean[i]), r: 4.5, fill: METHODS[m].color, stroke: "#fff", "stroke-width": 2 }));
        const html = `<div class="tt-title">t = ${(i * 0.04).toFixed(2)} s${i >= 3 && i <= 7 ? " · contact" : ""}</div>` +
          ms.map((m) => `<div class="tt-row"><span class="swatch" style="--c:${METHODS[m].color}"></span>${nm(m)}<span class="v">${RW[m].mean[i].toFixed(3)}</span></div>`).join("");
        const hb = host.getBoundingClientRect();
        tt.show(html, (x(i) / W) * b.width + (b.left - hb.left), ev.clientY - hb.top);
      };
      overlay.addEventListener("mousemove", move);
      overlay.addEventListener("mouseleave", () => { tt.hide(); cross.setAttribute("opacity", 0); dots.innerHTML = ""; });
      host.querySelector("svg")?.remove();
      host.prepend(s);
    }
    draw();
    onResize(host, draw);
  }

  // ------------------------------------------------------------------ latent vs. physical distances (hexbins)
  // Hexagons extracted from the paper's vector figure by tools/extract_rsa_hexbins.py.
  async function initRsa() {
    const grid = $("#rsa-grid");
    if (!grid) return;
    let data;
    try { data = await (await fetch("static/data/rsa_hexbins.json")).json(); }
    catch (e) { grid.textContent = "Could not load the plot data."; return; }
    // single-hue sequential ramp (light -> dark blue), density is relative per panel
    const STOPS = [[0, [234, 242, 250]], [0.35, [158, 197, 229]], [0.65, [59, 135, 196]], [1, [11, 53, 99]]];
    const ramp = (t) => {
      let k = 1;
      while (k < STOPS.length - 1 && t > STOPS[k][0]) k++;
      const [t0, c0] = STOPS[k - 1], [t1, c1] = STOPS[k];
      const u = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
      return `rgb(${c0.map((c, i) => Math.round(c + (c1[i] - c) * u)).join(",")})`;
    };
    const W = 240, H = 196, L = 34, R = 8, T = 8, B = 36, XMAX = 3.35, YMAX = 2.4;
    const sx = (W - L - R) / XMAX, sy = (H - T - B) / YMAX;
    const X = (v) => L + v * sx, Y = (v) => H - B - v * sy;
    data.panels.forEach((cells, k) => {
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${data.names[k]}: distance in latent space against distance in physical parameters` });
      const clipId = `rsa-clip-${k}`;
      const defs = svg("defs");
      const cp = svg("clipPath", { id: clipId });
      cp.append(svg("rect", { x: L, y: T, width: W - L - R, height: H - T - B }));
      defs.append(cp);
      s.append(defs);
      const g = svg("g", { class: "grid" });
      for (let v = 0; v <= 3; v++) {
        g.append(svg("line", { x1: X(v), x2: X(v), y1: T, y2: H - B }));
        const t = svg("text", { x: X(v), y: H - B + 14, "text-anchor": "middle", class: "tick-label" });
        t.textContent = v;
        g.append(t);
      }
      for (let v = 0; v <= 2; v++) {
        g.append(svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v) }));
        const t = svg("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "tick-label" });
        t.textContent = v;
        g.append(t);
      }
      s.append(g);
      const hexes = svg("g", { "clip-path": `url(#${clipId})` });
      const sorted = [...cells].sort((a, b) => a[2] - b[2]);
      for (const [cx, cy, t] of sorted) {
        const pts = data.hexagon.map(([dx, dy]) => `${(X(cx + dx)).toFixed(1)},${(Y(cy + dy)).toFixed(1)}`).join(" ");
        hexes.append(svg("polygon", { points: pts, fill: ramp(t), stroke: ramp(t), "stroke-width": 0.4 }));
      }
      s.append(hexes);
      s.append(svg("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "#d9d2cb" }));
      const xl = svg("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "axis-title" });
      xl.textContent = "distance in ρ (physical)";
      const yl = svg("text", { x: 10, y: (T + H - B) / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 10 ${(T + H - B) / 2})` });
      yl.textContent = "distance in r (latent)";
      s.append(xl, yl);
      grid.append(el("figure", { class: "rsa-panel" }, [
        el("h4", {}, [data.names[k] + " ", el("span", { class: "rsa-stat", text: `Spearman ${data.spearman[k].toFixed(2)}` })]),
        s,
      ]));
    });
  }

  // ------------------------------------------------------------------ small widgets
  function initFlipbook() {
    const fb = $("#flipbook");
    if (!fb) return;
    const v = fb.querySelector("video");
    whenVisible(fb, (vis) => (vis ? v.play().catch(() => {}) : v.pause()), { threshold: 0.1 });
  }

  function initLatentTabs() {
    const tabs = $("#latent-tabs");
    if (!tabs) return;
    const img = $("#latent-img"), cap = $("#latent-cap");
    tabs.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      img.src = b.dataset.img;
      cap.textContent = b.dataset.cap;
    }));
  }

  function initBibtex() {
    const btn = $("#copy-bib");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const text = $("#bibtex-code").textContent;
      try { await navigator.clipboard.writeText(text); }
      catch {
        const ta = el("textarea"); ta.value = text; document.body.append(ta); ta.select();
        document.execCommand("copy"); ta.remove();
      }
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = "Copy"), 1800);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initReel();
    initPlayer();
    initMseChart();
    initRwChart();
    initFlipbook();
    initLatentTabs();
    initRsa();
    initBibtex();
  });
})();
