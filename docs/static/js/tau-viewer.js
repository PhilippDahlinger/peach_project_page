/* 3D view of the tau explainer: space-time patches of a deforming 1D profile.
 *
 * Axes: x = position along the profile, h = deformation, tau * t = scaled time.
 * Patches, centers and colors are precomputed by tools/tau_patches.py (FPS in (x, h, tau*t)
 * for every slider step, colors matched between steps so that few points change color);
 * this file only displays docs/static/data/tau_patches.json.
 */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "./vendor/CSS2DRenderer.js";

const host = document.getElementById("tau-3d");
const slider = document.getElementById("tau-slider");
const out = document.getElementById("tau-value");
const hint = document.getElementById("tau-hint");

// world coordinates: X = x - 0.5 (profile), Y = h (deformation, up), Z = -tau * t (time, away from viewer)
const toWorld = (x, h, zt) => new THREE.Vector3(x - 0.5, h, -zt);
// look along the time axis from slightly above and to the right: time recedes into depth
const VIEW_DIR = new THREE.Vector3(0.42, 0.5, 1.0).normalize();

function label(html, cls) {
  const div = document.createElement("div");
  div.className = cls;
  div.innerHTML = html;
  return new CSS2DObject(div);
}

function hintFor(step) {
  if (step.singleFrame) return "<b>Large \u03c4: one frame per patch.</b>";
  if (step.framesPerPatch > 2.6) return "<b>Small \u03c4: patches mix many frames.</b>";
  return "<b>Intermediate \u03c4: patches are local in space and time.</b>";
}

async function init() {
  const status = host.querySelector(".vp-status");
  let data;
  try {
    data = await (await fetch("static/data/tau_patches.json")).json();
  } catch (e) {
    status.textContent = "Could not load the patch data.";
    return;
  }
  const { points, steps, palette, nFrames: F, dip } = data;
  const N = points.x.length;
  const prof = (x, t) => -dip * t * Math.exp(-((x - 0.5) ** 2) / 0.03);
  const h = points.x.map((x, i) => prof(x, points.t[i]));
  const colors = palette.map((c) => new THREE.Color(c));

  // ---------------------------------------------------------------- renderers
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    status.textContent = "WebGL is not available in this browser.";
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.append(renderer.domElement);
  const labels = new CSS2DRenderer();
  labels.domElement.className = "tau3d-labels";
  host.append(labels.domElement);
  status.textContent = "";

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd8cfc7, 1.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.3);
  sun.position.set(1, 2, 1.5);
  scene.add(sun);

  const camera = new THREE.PerspectiveCamera(32, 4 / 3, 0.01, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 0.3;
  controls.maxDistance = 40;

  // ---------------------------------------------------------------- points and centers
  const pointGeo = new THREE.SphereGeometry(0.016, 16, 12);
  const pointMesh = new THREE.InstancedMesh(pointGeo, new THREE.MeshStandardMaterial({ roughness: 0.45 }), N);
  const centerGeo = new THREE.SphereGeometry(0.03, 20, 16);
  const centerMesh = new THREE.InstancedMesh(centerGeo, new THREE.MeshStandardMaterial({ roughness: 0.4 }), steps[0].centers.length);
  const ringMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.037, 20, 16),
    new THREE.MeshBasicMaterial({ color: 0x1f2430, side: THREE.BackSide }), steps[0].centers.length);
  scene.add(pointMesh, centerMesh, ringMesh);

  // one faint line per frame showing the true profile
  const lineMat = new THREE.LineBasicMaterial({ color: 0x8a7d74, transparent: true, opacity: 0.35 });
  const frameLines = [];
  for (let f = 0; f < F; f++) {
    const geo = new THREE.BufferGeometry().setFromPoints(new Array(101).fill(0).map(() => new THREE.Vector3()));
    const line = new THREE.Line(geo, lineMat);
    frameLines.push(line);
    scene.add(line);
  }

  // ---------------------------------------------------------------- coordinate frame
  const O = toWorld(-0.08, 0, -0.1); // origin in front of frame 1, left of the profile
  const axisX = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), O, 1.22, 0xd62728, 0.05, 0.028);
  const axisY = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), O.clone().setY(-dip - 0.06), dip + 0.28, 0x2ca02c, 0.05, 0.028);
  const axisZ = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), O, 1, 0x1f77b4, 0.05, 0.028);
  scene.add(axisX, axisY, axisZ);
  const lx = label('x <span class="sub">position along the profile</span>', "axis-label x");
  const ly = label('h <span class="sub">deformation</span>', "axis-label y");
  const lz = label('τ·t <span class="sub">scaled time</span>', "axis-label z");
  lx.position.copy(O).add(new THREE.Vector3(1.3, 0, 0));
  ly.position.copy(O).add(new THREE.Vector3(0.13, 0, 0)).setY(0.25);
  scene.add(lx, ly, lz);
  const t0 = label("frame 1", "tick-label");
  const t1 = label(`frame ${F}`, "tick-label");
  scene.add(t0, t1);

  // ---------------------------------------------------------------- update per slider step
  const m = new THREE.Matrix4();
  let current = steps[0];
  function setStep(k) {
    const step = steps[k];
    current = step;
    const tau = step.tau;
    out.textContent = tau < 1 ? tau.toFixed(2) : tau.toFixed(1);
    for (let i = 0; i < N; i++) {
      m.makeTranslation(toWorld(points.x[i], h[i], tau * points.t[i]));
      pointMesh.setMatrixAt(i, m);
      pointMesh.setColorAt(i, colors[parseInt(step.colors[i], 36)]);
    }
    step.centers.forEach((c, j) => {
      m.makeTranslation(toWorld(points.x[c], h[c], tau * points.t[c]));
      centerMesh.setMatrixAt(j, m);
      ringMesh.setMatrixAt(j, m);
      centerMesh.setColorAt(j, colors[parseInt(step.colors[c], 36)]);
    });
    for (const im of [pointMesh, centerMesh, ringMesh]) {
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
      im.computeBoundingSphere();
    }
    frameLines.forEach((line, f) => {
      const t = f / (F - 1);
      const pos = line.geometry.attributes.position;
      for (let i = 0; i <= 100; i++) {
        const x = i / 100;
        const p = toWorld(x, prof(x, t), tau * t);
        pos.setXYZ(i, p.x, p.y, p.z);
      }
      pos.needsUpdate = true;
      line.geometry.computeBoundingSphere();
    });
    axisZ.setLength(tau + 0.35, 0.05, 0.028);
    lz.position.copy(O).add(new THREE.Vector3(0, 0, -(tau + 0.45)));
    t0.position.copy(toWorld(1.1, 0.02, 0));
    t1.position.copy(toWorld(1.1, 0.02, tau));
    t1.visible = tau > 0.35;
    hint.innerHTML = hintFor(step);
    fit(false);
  }

  // keep the current viewing direction and frame the scene. Only the near part of a long time
  // axis is framed; farther frames recede into the distance.
  function fit(resetDirection) {
    const tau = current.tau;
    const box = new THREE.Box3(toWorld(-0.15, -dip - 0.06, Math.min(tau, 2.2) + 0.3), toWorld(1.3, 0.3, -0.12));
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() / 2;
    const dir = resetDirection ? VIEW_DIR.clone() : camera.position.clone().sub(controls.target).normalize();
    const vfov = THREE.MathUtils.degToRad(camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
    const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 0.8;
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(dir, dist);
    controls.update();
  }

  function resize() {
    const w = host.clientWidth, hh = host.clientHeight;
    renderer.setSize(w, hh);
    labels.setSize(w, hh);
    camera.aspect = w / hh;
    camera.updateProjectionMatrix();
  }

  resize();
  slider.max = steps.length - 1;
  const fillRange = window.PEACH_syncRangeFill ? window.PEACH_syncRangeFill(slider) : () => {};
  slider.addEventListener("input", () => { fillRange(); setStep(+slider.value); });
  setStep(+slider.value);
  fit(true);
  document.getElementById("tau-reset").addEventListener("click", () => fit(true));
  new ResizeObserver(() => { resize(); fit(false); }).observe(host);

  let visible = true;
  new IntersectionObserver((es) => es.forEach((e) => (visible = e.isIntersecting))).observe(host);
  (function loop() {
    requestAnimationFrame(loop);
    if (!visible) return;
    controls.update();
    renderer.render(scene, camera);
    labels.render(scene, camera);
  })();
}

if (host) {
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); init(); }
  }, { rootMargin: "400px" });
  io.observe(host);
}
