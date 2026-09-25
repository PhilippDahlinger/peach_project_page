/* Side-by-side 3D viewer: predicted mesh trajectory vs. observed point cloud.
 *
 * Every .viewport[data-scene] loads a scene written by tools/realworld_scene.py
 * (<name>.json + <name>.bin). All viewports share one camera pose and one timeline.
 */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const root = document.getElementById("viewer3d");

const MESH_COLORS = { "PEACH (ours)": 0x2f7fb8, "No Context": 0x8d8d8d };
const POINT_COLORS = [0xd62728, 0x8b0f10]; // sheet, ball
const DTYPES = { float32: Float32Array, uint32: Uint32Array, uint8: Uint8Array };

async function loadScene(url) {
  const meta = await (await fetch(url)).json();
  const binUrl = new URL(meta.bin, new URL(url, location.href));
  const buf = await (await fetch(binUrl)).arrayBuffer();
  const get = (k) => {
    const b = meta.buffers[k];
    return new DTYPES[b.dtype](buf, b.offset, b.length);
  };
  return {
    meta,
    meshPositions: get("meshPositions"),
    pointPositions: get("pointPositions"),
    meshIndex: get("meshIndex"),
    meshObjectId: get("meshObjectId"),
    pointObjectId: get("pointObjectId"),
  };
}

function discTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(28, 26, 4, 32, 32, 30);
  grd.addColorStop(0, "#ffffff");
  grd.addColorStop(0.25, "#ffffff");
  grd.addColorStop(1, "#bbbbbb");
  g.fillStyle = grd;
  g.beginPath();
  g.arc(32, 32, 29, 0, Math.PI * 2);
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

class SceneView {
  constructor(el, data, sharedCamera) {
    this.el = el;
    this.data = data;
    const { meta } = data;
    this.T = meta.frameTimes.length;
    this.N = meta.mesh.numVertices;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.append(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = sharedCamera;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xd9cfc6, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1.5, -2, 3);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-2, 1.5, 1);
    this.scene.add(fill);

    // normalize: center on the bounding box, longest side = 1
    const [lo, hi] = meta.bounds;
    const center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
    const scale = 1 / Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    this.world = new THREE.Group();
    this.world.scale.setScalar(scale);
    this.world.position.copy(center).multiplyScalar(-scale);
    this.scene.add(this.world);

    // mesh
    const base = MESH_COLORS[meta.label] ?? 0x2f7fb8;
    const colors = new Float32Array(this.N * 3);
    const cSheet = new THREE.Color(base), cBall = new THREE.Color(base).offsetHSL(0, 0, -0.12);
    for (let i = 0; i < this.N; i++) (data.meshObjectId[i] ? cBall : cSheet).toArray(colors, i * 3);
    this.meshGeom = new THREE.BufferGeometry();
    this.meshGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.N * 3), 3));
    this.meshGeom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.meshGeom.setIndex(new THREE.BufferAttribute(data.meshIndex, 1));
    this.mesh = new THREE.Mesh(this.meshGeom, new THREE.MeshStandardMaterial({
      vertexColors: true, side: THREE.DoubleSide, roughness: 0.55, metalness: 0.05,
      transparent: true, opacity: 0.62, depthWrite: false, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    }));
    this.wire = new THREE.Mesh(this.meshGeom, new THREE.MeshBasicMaterial({
      color: new THREE.Color(base).offsetHSL(0, 0, -0.25), wireframe: true, transparent: true, opacity: 0.35,
    }));
    this.wire.visible = false;
    this.world.add(this.mesh, this.wire);

    // points
    const maxPts = Math.max(...meta.points.frameOffsets.slice(1).map((o, i) => o - meta.points.frameOffsets[i]));
    this.ptsGeom = new THREE.BufferGeometry();
    this.ptsGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(maxPts * 3), 3));
    this.ptsGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(maxPts * 3), 3));
    this.points = new THREE.Points(this.ptsGeom, new THREE.PointsMaterial({
      size: 0.017, vertexColors: true, map: discTexture(), alphaTest: 0.5, sizeAttenuation: true,
    }));
    this.world.add(this.points);

    this.resize();
    this.setFrame(0);
  }

  setFrame(k) {
    const { data, N } = this;
    const pos = this.meshGeom.attributes.position;
    pos.array.set(data.meshPositions.subarray(k * N * 3, (k + 1) * N * 3));
    pos.needsUpdate = true;
    this.meshGeom.computeVertexNormals();
    this.meshGeom.computeBoundingSphere();

    const off = data.meta.points.frameOffsets;
    const a = off[k], b = off[k + 1];
    const pp = this.ptsGeom.attributes.position, pc = this.ptsGeom.attributes.color;
    pp.array.set(data.pointPositions.subarray(a * 3, b * 3));
    const cols = POINT_COLORS.map((c) => new THREE.Color(c));
    for (let i = a; i < b; i++) cols[Math.min(data.pointObjectId[i], 1)].toArray(pc.array, (i - a) * 3);
    pp.needsUpdate = pc.needsUpdate = true;
    this.ptsGeom.setDrawRange(0, b - a);
    this.ptsGeom.computeBoundingSphere();
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.renderer.setSize(w, h, false);
    this.aspect = w / h;
  }

  render() {
    this.camera.aspect = this.aspect;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
  }
}

async function init() {
  const vpEls = [...root.querySelectorAll(".viewport[data-scene]")];
  const camera = new THREE.PerspectiveCamera(35, 4 / 3, 0.01, 50);
  camera.up.set(0, 0, 1);
  const HOME = { pos: new THREE.Vector3(1.0, -1.5, 0.5), target: new THREE.Vector3(0, 0, -0.16) };
  camera.position.copy(HOME.pos);

  const results = await Promise.allSettled(vpEls.map((v) => loadScene(v.dataset.scene)));
  const views = [];
  results.forEach((r, i) => {
    const status = vpEls[i].querySelector(".vp-status");
    if (r.status === "fulfilled") {
      views.push(new SceneView(vpEls[i], r.value, camera));
      status.textContent = "";
      const name = vpEls[i].querySelector(".vp-name");
      if (r.value.meta.label) name.textContent = r.value.meta.label;
    } else {
      status.textContent = "Could not load scene data.";
      console.error(r.reason);
    }
  });
  if (!views.length) return;
  if (!views.some((v) => v.data.meta.meta?.placeholder)) document.getElementById("viewer-banner")?.remove();

  // one OrbitControls per canvas, all driving the shared camera
  const controls = views.map((v) => {
    const c = new OrbitControls(camera, v.renderer.domElement);
    c.target.copy(HOME.target);
    c.enableDamping = true;
    c.dampingFactor = 0.12;
    c.minDistance = 0.4;
    c.maxDistance = 6;
    return c;
  });
  const syncTargets = (src) => controls.forEach((c) => c !== src && c.target.copy(src.target));
  controls.forEach((c) => c.addEventListener("change", () => syncTargets(c)));

  // timeline
  const T = Math.min(...views.map((v) => v.T));
  const times = views[0].data.meta.frameTimes;
  const scrub = document.getElementById("v3d-scrub");
  const playBtn = document.getElementById("v3d-play");
  const tLabel = document.getElementById("v3d-time");
  scrub.max = T - 1;
  const fill = window.PEACH_syncRangeFill ? window.PEACH_syncRangeFill(scrub) : () => {};
  let frame = Math.min(+scrub.value, T - 1), playing = false, last = 0, visible = false;
  const setFrame = (k) => {
    frame = k;
    views.forEach((v) => v.setFrame(k));
    scrub.value = k;
    fill();
    tLabel.textContent = `t = ${times[k].toFixed(3)} s`;
  };
  const setPlaying = (p) => {
    playing = p;
    playBtn.classList.toggle("paused", !p);
    playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
  };
  setPlaying(false);
  setFrame(frame);
  scrub.addEventListener("input", () => { setPlaying(false); setFrame(+scrub.value); });
  playBtn.addEventListener("click", () => setPlaying(!playing));

  document.getElementById("v3d-mesh").addEventListener("change", (e) => views.forEach((v) => (v.mesh.visible = e.target.checked)));
  document.getElementById("v3d-wire").addEventListener("change", (e) => views.forEach((v) => (v.wire.visible = e.target.checked)));
  document.getElementById("v3d-points").addEventListener("change", (e) => views.forEach((v) => (v.points.visible = e.target.checked)));
  document.getElementById("v3d-reset").addEventListener("click", () => {
    camera.position.copy(HOME.pos);
    controls.forEach((c) => { c.target.copy(HOME.target); c.update(); });
  });

  new ResizeObserver(() => views.forEach((v) => v.resize())).observe(vpEls[0]);
  new IntersectionObserver((es) => es.forEach((e) => (visible = e.isIntersecting)), { rootMargin: "100px" }).observe(root);

  const frameMs = 1000 * (times[1] - times[0] || 1 / 30) * 3; // 3x slow motion
  (function loop(now) {
    requestAnimationFrame(loop);
    if (!visible) return;
    if (playing && now - last > frameMs) {
      last = now;
      setFrame((frame + 1) % T);
    }
    controls.forEach((c) => c.update());
    views.forEach((v) => v.render());
  })(0);
}

if (root) {
  // defer loading until the viewer is close to the viewport
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); init(); }
  }, { rootMargin: "400px" });
  io.observe(root);
}
