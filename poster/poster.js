/* Vector charts for the A0 poster. SVG units are millimeters (viewBox = printed size). */
(() => {
  "use strict";
  const PT = 0.3528; // 1 pt in mm
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const pow10 = (e) => `10${String(e).split("").map((c) => SUP[c]).join("")}`;
  const INK = "#1f2430", INK2 = "#444b5a", INK3 = "#737a88", GRID = "#efe6de";

  const METHODS = {
    peach: ["PEACH (ours)", "#1f77b4"], pstnet: ["PSTNet Encoder", "#e377c2"], gnn: ["GNN Encoder", "#8c564b"],
    tto: ["Test-Time Opt.", "#bcbd22"], mango: ["MaNGO (mesh context)", "#17becf"], oracle: ["Oracle", "#2ca02c"],
    oracle_mgn: ["Oracle (MGN)", "#9467bd"], nocontext: ["No Context", "#7f7f7f"], nocontext_mgn: ["No Context (MGN)", "#ff7f0e"],
    peach_noaug: ["PEACH (no data augm.)", "#9467bd"],
  };
  const ORDER = ["peach", "pstnet", "gnn", "tto", "mango", "oracle", "oracle_mgn", "nocontext", "nocontext_mgn"];
  const GROUP_GAP_AFTER = new Set(["tto", "mango", "oracle_mgn"]);
  const MSE = {
    "Deforming Block": { peach: [1.4707e-6, 1.3136e-6, 1.6826e-6], pstnet: [8.7298e-6, 5.6903e-6, 1.1975e-5], gnn: [1.9537e-5, 4.8388e-6, 4.4339e-5], tto: [1.2119e-5, 1.0742e-5, 1.3495e-5], mango: [1.2112e-6, 1.1531e-6, 1.2742e-6], oracle: [1.2985e-6, 1.1952e-6, 1.4497e-6], oracle_mgn: [4.1116e-6, 3.7078e-6, 4.5653e-6], nocontext: [5.0834e-5, 5.0709e-5, 5.0959e-5], nocontext_mgn: [5.7853e-5, 5.6436e-5, 5.9399e-5] },
    "Sheet Deformation": { peach: [1.45e-7, 1.22e-7, 1.70e-7], pstnet: [2.65e-5, 2.64e-5, 2.66e-5], gnn: [1.49e-6, 6.23e-7, 2.43e-6], tto: [4.70e-8, 4.49e-8, 5.00e-8], mango: [3.89e-7, 3.34e-7, 4.44e-7], oracle: [4.52e-8, 4.23e-8, 4.81e-8], oracle_mgn: [7.92e-7, 6.98e-7, 9.29e-7], nocontext: [2.66e-5, 2.64e-5, 2.69e-5], nocontext_mgn: [2.90e-5, 2.85e-5, 2.94e-5] },
    "Bending Beam": { peach: [2.68e-4, 2.61e-4, 2.74e-4], pstnet: [1.20e-3, 9.16e-4, 1.51e-3], gnn: [8.70e-4, 6.82e-4, 9.86e-4], tto: [5.52e-5, 5.13e-5, 5.90e-5], mango: [3.64e-4, 3.31e-4, 4.03e-4], oracle: [2.93e-4, 2.80e-4, 3.07e-4], oracle_mgn: [1.85e-3, 1.55e-3, 2.12e-3], nocontext: [7.89e-4, 7.70e-4, 8.12e-4], nocontext_mgn: [2.04e-3, 1.70e-3, 2.33e-3] },
    "Trampoline": { peach: [1.1675e-5, 1.0020e-5, 1.2944e-5], pstnet: [5.3091e-5, 4.4918e-5, 6.4341e-5], gnn: [3.1270e-4, 2.8242e-4, 3.3119e-4], tto: [1.2343e-5, 1.1000e-5, 1.4248e-5], mango: [2.0718e-5, 1.7016e-5, 2.4613e-5], oracle: [4.3457e-6, 4.1519e-6, 4.4844e-6], oracle_mgn: [1.4253e-4, 1.3208e-4, 1.5335e-4], nocontext: [2.9218e-4, 2.8797e-4, 2.9636e-4], nocontext_mgn: [8.5081e-4, 7.3378e-4, 9.8727e-4] },
  };
  const RW = {
    peach: { mean: [0.00517, 0.01245, 0.03685, 0.16925, 0.46502, 0.47792, 0.58516, 0.35339, 0.3073, 0.37552, 0.51994], lo: [0.00503, 0.01231, 0.03646, 0.16702, 0.45493, 0.45458, 0.52323, 0.33003, 0.2942, 0.34946, 0.49163], hi: [0.00533, 0.01264, 0.03719, 0.17148, 0.47511, 0.49747, 0.64712, 0.38011, 0.32722, 0.4011, 0.54824] },
    peach_noaug: { mean: [0.00536, 0.01261, 0.03682, 0.16911, 0.46456, 0.50467, 0.70864, 0.41338, 0.30866, 0.3643, 0.50754], lo: [0.00502, 0.01228, 0.03658, 0.16706, 0.44092, 0.48404, 0.66896, 0.40547, 0.30254, 0.36018, 0.49419], hi: [0.00572, 0.01299, 0.03725, 0.17039, 0.47987, 0.52529, 0.73602, 0.42277, 0.31454, 0.36841, 0.5209] },
    oracle: { mean: [0.00502, 0.01227, 0.03663, 0.17186, 0.53636, 0.86493, 0.75647, 0.47337, 0.47714, 0.51604, 0.58717], lo: [0.00501, 0.01224, 0.0365, 0.17015, 0.53206, 0.85425, 0.73765, 0.45989, 0.4716, 0.51268, 0.58334], hi: [0.00503, 0.0123, 0.03672, 0.17357, 0.54405, 0.87941, 0.77891, 0.48684, 0.4814, 0.51941, 0.59102] },
    tto: { mean: [0.00636, 0.01474, 0.03906, 0.18832, 0.52431, 0.97206, 0.97042, 0.52407, 0.42635, 0.4342, 0.5396], lo: [0.00553, 0.01284, 0.03317, 0.12761, 0.44633, 0.83412, 0.85703, 0.37808, 0.37554, 0.40252, 0.4966], hi: [0.00756, 0.01771, 0.04272, 0.24406, 0.59393, 1.11, 1.04278, 0.72817, 0.47715, 0.47157, 0.5826] },
    nocontext: { mean: [0.00617, 0.01336, 0.03793, 0.17288, 0.55071, 1.16768, 1.16615, 0.49832, 0.3802, 0.44616, 0.56058], lo: [0.00512, 0.01245, 0.03704, 0.17009, 0.53873, 1.11711, 1.154, 0.49394, 0.36324, 0.43007, 0.54187], hi: [0.00812, 0.01489, 0.03947, 0.17557, 0.56269, 1.20231, 1.17717, 0.50269, 0.39715, 0.46211, 0.57937] },
    nocontext_mgn: { mean: [0.00526, 0.01322, 0.04098, 0.16844, 0.52522, 1.26656, 1.38343, 0.76764, 0.78091, 0.96148, 1.0907], lo: [0.00513, 0.01269, 0.0388, 0.16726, 0.48638, 1.01408, 1.32488, 0.61874, 0.59531, 0.6982, 0.74745], hi: [0.00541, 0.01375, 0.04352, 0.16992, 0.57129, 1.5753, 1.48175, 1.02241, 1.07759, 1.38303, 1.61138] },
  };

  const fs = (pt) => (pt * PT).toFixed(2);
  const makeSvg = (host, W, H) => {
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: `${W}mm`, height: `${H}mm` });
    host.append(s);
    return s;
  };
  const text = (s, x, y, str, pt, attrs = {}) => s.append(svg("text", { x, y, "font-size": fs(pt), fill: INK2, ...attrs }, str));

  // ---------------------------------------------------------------- MSE small multiples
  function mse() {
    const host = document.getElementById("mse-charts");
    const legend = document.getElementById("mse-legend");
    for (const m of ORDER) {
      const sp = document.createElement("span");
      sp.innerHTML = `<i style="background:${METHODS[m][1]}"></i>${METHODS[m][0]}`;
      legend.append(sp);
    }
    const W = host.clientWidth / (96 / 25.4) / 2 - 3; // mm per panel
    const H = 66, L = 17, R = 2, T = 10, B = 3;
    for (const [scene, data] of Object.entries(MSE)) {
      const box = document.createElement("div");
      host.append(box);
      const s = makeSvg(box, W, H);
      const all = ORDER.flatMap((m) => [data[m][1], data[m][2]]);
      const e0 = Math.floor(Math.log10(Math.min(...all))), e1 = Math.ceil(Math.log10(Math.max(...all)));
      const y = (v) => T + (1 - (Math.log10(v) - e0) / (e1 - e0)) * (H - T - B);
      text(s, L, 6.3, scene, 19, { fill: INK, "font-weight": 650 });
      for (let e = e0; e <= e1; e++) {
        s.append(svg("line", { x1: L, x2: W - R, y1: y(10 ** e), y2: y(10 ** e), stroke: GRID, "stroke-width": 0.3 }));
        text(s, L - 1.5, y(10 ** e) + 1.8, pow10(e), 14, { "text-anchor": "end", fill: INK3 });
      }
      const slots = ORDER.length + GROUP_GAP_AFTER.size * 0.6;
      const bw = (W - L - R - 2) / slots;
      let xi = 1;
      for (const m of ORDER) {
        const [mean, lo, hi] = data[m];
        const x0 = L + xi * bw - bw * 0.5, w = bw * 0.78;
        const yb = H - B, yt = y(mean);
        s.append(svg("rect", { x: x0, y: yt, width: w, height: yb - yt, fill: METHODS[m][1], rx: 0.6 }));
        const cx = x0 + w / 2;
        s.append(svg("line", { x1: cx, x2: cx, y1: y(hi), y2: y(lo), stroke: INK, "stroke-width": 0.35 }));
        s.append(svg("line", { x1: cx - w * 0.25, x2: cx + w * 0.25, y1: y(hi), y2: y(hi), stroke: INK, "stroke-width": 0.35 }));
        s.append(svg("line", { x1: cx - w * 0.25, x2: cx + w * 0.25, y1: y(lo), y2: y(lo), stroke: INK, "stroke-width": 0.35 }));
        xi += 1 + (GROUP_GAP_AFTER.has(m) ? 0.6 : 0);
      }
      s.append(svg("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: "#c9bfb6", "stroke-width": 0.35 }));
    }
  }

  // ---------------------------------------------------------------- real-world line chart
  function rwChart() {
    const host = document.getElementById("rw-chart");
    const W = host.clientWidth / (96 / 25.4), H = W * 0.68;
    const L = 17, R = 7, T = 13, B = 13, N = 11, ymax = 1.6;
    const s = makeSvg(host, W, H);
    const x = (i) => L + (i / (N - 1)) * (W - L - R), y = (v) => T + (1 - v / ymax) * (H - T - B);
    s.append(svg("rect", { x: x(3), y: T, width: x(7) - x(3), height: H - T - B, fill: "#fdecea" }));
    text(s, (x(3) + x(7)) / 2, T + 6, "ball–sheet contact", 15, { "text-anchor": "middle", fill: "#b83227", "font-weight": 650 });
    for (let v = 0; v <= 1.5001; v += 0.5) {
      s.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: GRID, "stroke-width": 0.3 }));
      text(s, L - 1.8, y(v) + 1.8, v.toFixed(1), 14, { "text-anchor": "end", fill: INK3 });
    }
    for (let i = 0; i < N; i += 2) text(s, x(i), H - B + 6, (i * 0.04).toFixed(2), 14, { "text-anchor": "middle", fill: INK3 });
    text(s, (L + W - R) / 2, H - 1.5, "time [s]", 15, { "text-anchor": "middle", fill: INK3 });
    text(s, 4.5, (T + H - B) / 2, "distance [×10⁻³]", 15, { "text-anchor": "middle", fill: INK3, transform: `rotate(-90 4.5 ${(T + H - B) / 2})` });
    const order = ["nocontext_mgn", "nocontext", "tto", "oracle", "peach_noaug", "peach"];
    for (const m of order) {
      const d = RW[m];
      const band = d.hi.map((v, i) => `${x(i)},${y(v)}`).concat(d.lo.map((v, i) => `${x(i)},${y(v)}`).reverse());
      s.append(svg("polygon", { points: band.join(" "), fill: METHODS[m][1], opacity: 0.13 }));
    }
    for (const m of order) {
      s.append(svg("polyline", { points: RW[m].mean.map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: METHODS[m][1], "stroke-width": m === "peach" ? 1.3 : 0.8, "stroke-linejoin": "round" }));
    }
    const leg = document.createElement("div");
    leg.className = "legend rw-legend";
    [["peach", "PEACH (ours)"], ["peach_noaug", "PEACH, no data augm."], ["oracle", "\u201cOracle\u201d"], ["tto", "Test-Time Opt."], ["nocontext", "No Context"], ["nocontext_mgn", "No Context (MGN)"]]
      .forEach(([m, name]) => { const sp = document.createElement("span"); sp.innerHTML = `<i style="background:${METHODS[m][1]}"></i>${name}`; leg.append(sp); });
    host.append(leg);
  }

  // ---------------------------------------------------------------- runtime bars
  function runtime() {
    const host = document.getElementById("runtime-chart");
    const W = host.clientWidth / (96 / 25.4), H = 36, L = 58, R = 22;
    const s = makeSvg(host, W, H);
    const X = (v) => L + (Math.log10(v) + 1) / 4 * (W - L - R);
    for (const t of [0.1, 1, 10, 100, 1000]) {
      s.append(svg("line", { x1: X(t), x2: X(t), y1: 0, y2: H - 7, stroke: GRID, "stroke-width": 0.3 }));
      text(s, X(t), H - 1.5, t === 1000 ? "1000 s" : String(t), 13, { "text-anchor": "middle", fill: INK3 });
    }
    [["PEACH (ours)", 0.581, "0.58 s", "#1f77b4"], ["Test-Time Opt.", 81.93, "82 s", "#bcbd22"], ["FEM (forward)", 240, "240 s", "#7f7f7f"]].forEach(([n, v, vl, c], k) => {
      const yy = 1.5 + k * 9.2;
      text(s, 0, yy + 5.2, n, 16, { fill: INK, "font-weight": k === 0 ? 700 : 400 });
      s.append(svg("rect", { x: X(0.1), y: yy, width: X(v) - X(0.1), height: 6.5, rx: 1, fill: c }));
      text(s, X(v) + 2, yy + 5.2, vl, 15, { fill: INK2, "font-family": "JetBrains Mono" });
    });
  }

  // ---------------------------------------------------------------- hexbins
  async function rsa() {
    const host = document.getElementById("rsa");
    const data = await (await fetch("../docs/static/data/rsa_hexbins.json")).json();
    const STOPS = [[0, [234, 242, 250]], [0.35, [158, 197, 229]], [0.65, [59, 135, 196]], [1, [11, 53, 99]]];
    const ramp = (t) => {
      let k = 1;
      while (k < STOPS.length - 1 && t > STOPS[k][0]) k++;
      const [t0, c0] = STOPS[k - 1], [t1, c1] = STOPS[k];
      const u = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
      return `rgb(${c0.map((c, i) => Math.round(c + (c1[i] - c) * u)).join(",")})`;
    };
    const W = host.clientWidth / (96 / 25.4) / 2 - 3.5, H = W * 0.44;
    const L = 11, R = 1.5, T = 2, B = 11, XMAX = 3.35, YMAX = 2.4;
    const X = (v) => L + v * (W - L - R) / XMAX, Y = (v) => H - B - v * (H - T - B) / YMAX;
    data.panels.forEach((cells, k) => {
      const fig = document.createElement("figure");
      fig.innerHTML = `<h4>${data.names[k]}<span>Spearman ${data.spearman[k].toFixed(2)}</span></h4>`;
      host.append(fig);
      const s = makeSvg(fig, W, H);
      const cp = svg("clipPath", { id: `c${k}` });
      cp.append(svg("rect", { x: L, y: T, width: W - L - R, height: H - T - B }));
      s.append(cp);
      for (let v = 0; v <= 3; v++) {
        s.append(svg("line", { x1: X(v), x2: X(v), y1: T, y2: H - B, stroke: GRID, "stroke-width": 0.3 }));
        text(s, X(v), H - B + 5, String(v), 13, { "text-anchor": "middle", fill: INK3 });
      }
      for (let v = 0; v <= 2; v++) {
        s.append(svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: GRID, "stroke-width": 0.3 }));
        text(s, L - 2, Y(v) + 1.6, String(v), 13, { "text-anchor": "end", fill: INK3 });
      }
      const g = svg("g", { "clip-path": `url(#c${k})` });
      for (const [cx, cy, t] of [...cells].sort((a, b) => a[2] - b[2])) {
        const pts = data.hexagon.map(([dx, dy]) => `${X(cx + dx).toFixed(2)},${Y(cy + dy).toFixed(2)}`).join(" ");
        g.append(svg("polygon", { points: pts, fill: ramp(t), stroke: ramp(t), "stroke-width": 0.15 }));
      }
      s.append(g);
      s.append(svg("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "#d9d2cb", "stroke-width": 0.3 }));
      text(s, (L + W - R) / 2, H - 1, "distance in ρ (physical)", 13.5, { "text-anchor": "middle", fill: INK3 });
      text(s, 3.2, (T + H - B) / 2, "distance in r (latent)", 13.5, { "text-anchor": "middle", fill: INK3, transform: `rotate(-90 3.2 ${(T + H - B) / 2})` });
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    await document.fonts.ready;
    // fit the wordmark SVG to its text
    const wm = document.querySelector(".wordmark svg"), wt = wm.querySelector("text");
    const w = wt.getBBox().width + 1;
    wm.setAttribute("viewBox", `0 0 ${w} 44`);
    wm.setAttribute("width", `${w}mm`);
    mse();
    rwChart();
    runtime();
    await rsa();
    document.body.dataset.ready = "1";
  });
})();
