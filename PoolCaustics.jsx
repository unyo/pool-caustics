import { useEffect, useRef } from "react";

/* ─────────────────────────────────────────────────────────────────────────────
 * PoolCaustics — pool-floor light with a touch of rainbow (WebGL2)
 *
 *   <PoolCaustics style={{ minHeight: "100vh" }}>
 *     ...your content...
 *   </PoolCaustics>
 *
 * How it works (same idea as the video):
 *   1. A moving sum of small ripples is the water surface (height + slope).
 *   2. Light is refracted through that surface with Snell's law. Where the
 *      surface focuses the rays, light piles up into bright sheets; the floor
 *      is just a slice through those sheets at `depth`.
 *   3. Water bends violet slightly more than red. We refract once per
 *      wavelength (IOR from a Cauchy fit: red ≈ 1.331, violet ≈ 1.343), splat
 *      each one onto the floor, and add them up. Where they overlap it is
 *      white; only the fringes split into a rainbow.  `dispersion` scales that
 *      spread: 1 = real water, 20 = the "exaggerated" look from the video.
 * ───────────────────────────────────────────────────────────────────────────── */

const NW = 16; // surface wave components
const MAX_S = 24; // max spectral samples (grows with `dispersion` to keep the rainbow smooth)
const PX_PER_METER = 750; // CSS px per metre of pool floor at zoom = 1
const TAU = Math.PI * 2;

const QUALITY = {
  low: { cellPx: 4, samples: 5, dpr: 1 },
  medium: { cellPx: 3, samples: 6, dpr: 1.25 },
  high: { cellPx: 2.5, samples: 8, dpr: 1.5 },
};

/* ── shaders ─────────────────────────────────────────────────────────────── */

// Fullscreen triangle, no buffers.
const VS_FULL = `#version 300 es
precision highp float;
void main(){
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Pass 1: evaluate the water surface once per mesh vertex -> (height, slopeX, slopeZ).
const FS_SURFACE = `#version 300 es
precision highp float;
uniform vec4 uWave[${NW}];      // kx, ky, amplitude, -
uniform float uPhase[${NW}];    // phase (time already folded in on the CPU)
uniform vec2 uSize;             // metres covered by the mesh
uniform vec2 uGrid;             // cells (nx, ny)
out vec4 o;
void main(){
  vec2 p = (floor(gl_FragCoord.xy) / uGrid - 0.5) * uSize;
  float h = 0.0; vec2 g = vec2(0.0);
  for (int i = 0; i < ${NW}; i++) {
    vec4 w = uWave[i];
    float ph = dot(w.xy, p) + uPhase[i];
    h += w.z * sin(ph);
    g += w.z * cos(ph) * w.xy;
  }
  o = vec4(h, g, 1.0);
}`;

// Pass 2: every vertex of a regular mesh on the water surface fires a refracted
// ray to the floor. One instance per wavelength (own IOR + RGB weight).
const VS_CAUSTIC = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uSurf;
uniform ivec2 uCells;
uniform vec2 uSize;
uniform vec2 uView;
uniform float uDepth;
uniform vec3 uLight;            // unit vector pointing toward the light
uniform float uIor[${MAX_S}];
uniform float uIor0;
uniform vec3 uWeight[${MAX_S}];
out vec2 vSrc;
out vec3 vW;
void main(){
  int stride = uCells.x + 1;
  ivec2 ij = ivec2(gl_VertexID % stride, gl_VertexID / stride);
  vec4 s = texelFetch(uSurf, ij, 0);
  vec2 p = (vec2(ij) / vec2(uCells) - 0.5) * uSize;

  vec3 N = normalize(vec3(-s.y, 1.0, -s.z));
  vec3 I = -uLight;
  vec3 R  = refract(I, N, 1.0 / uIor[gl_InstanceID]);
  vec3 R0 = refract(I, vec3(0.0, 1.0, 0.0), 1.0 / uIor0);

  vec2 hit  = p + R.xz  * ((-uDepth - s.x) / R.y);   // where this ray lands on the floor
  vec2 rest = R0.xz * (-uDepth / R0.y);              // where flat water would land it
  vec2 f = hit - rest;

  gl_Position = vec4(f / (uView * 0.5), 0.0, 1.0);
  vSrc = p;
  vW = uWeight[gl_InstanceID];
}`;

// Brightness = how much surface area is squeezed into one floor pixel.
const FS_CAUSTIC = `#version 300 es
precision highp float;
in vec2 vSrc;
in vec3 vW;
uniform float uPixArea;
uniform float uMaxI;
out vec4 o;
void main(){
  vec2 dx = dFdx(vSrc), dy = dFdy(vSrc);
  float I = abs(dx.x * dy.y - dx.y * dy.x) / uPixArea;
  o = vec4(vW * min(I, uMaxI), 1.0);
}`;

// Pass 3: tone-map to a dark floor + light, soft glow, film grain.
const FS_POST = `#version 300 es
precision highp float;
uniform sampler2D uAccum;
uniform vec2 uRes;
uniform vec3 uFloor;
uniform vec3 uLightC;
uniform float uGain;
uniform float uContrast;
uniform float uGrain;
uniform float uFrame;
out vec4 o;
float hash(vec2 p){
  vec3 p3 = fract(vec3(p.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 px = 1.0 / uRes;
  vec3 c = texture(uAccum, uv).rgb;
  vec3 near = (texture(uAccum, uv + px * vec2( .75,  .75)).rgb
             + texture(uAccum, uv + px * vec2(-.75,  .75)).rgb
             + texture(uAccum, uv + px * vec2( .75, -.75)).rgb
             + texture(uAccum, uv + px * vec2(-.75, -.75)).rgb) * .25;
  vec3 far  = (texture(uAccum, uv + px * vec2( 3., 0.)).rgb
             + texture(uAccum, uv + px * vec2(-3., 0.)).rgb
             + texture(uAccum, uv + px * vec2(0.,  3.)).rgb
             + texture(uAccum, uv + px * vec2(0., -3.)).rgb) * .25;
  c = mix(c, near, .3) + far * .06;

  vec3 t = pow(1.0 - exp(-c * uGain), vec3(uContrast));   // per-channel: hot cores burn to white
  vec3 col = pow(uFloor + uLightC * t, vec3(1.0 / 2.2));
  col += (hash(gl_FragCoord.xy + uFrame) - .5) * uGrain;
  o = vec4(col, 1.0);
}`;

/* ── helpers ─────────────────────────────────────────────────────────────── */

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Cauchy fit to water: n(550nm)=1.333, n(650)=1.3312, n(400)=1.3435 (λ in µm)
const waterIor = (um) => 1.3237 + 0.003168 / (um * um);

// Spectral samples (red→violet) as RGB weights that sum to exactly white per channel.
function spectralSamples(n) {
  const g = (l, mu, s1, s2) => { const t = (l - mu) / (l < mu ? s1 : s2); return Math.exp(-0.5 * t * t); };
  const rows = [], sum = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const nm = 680 - ((i + 0.5) * (680 - 420)) / n;
    const x = 1.056 * g(nm, 599.8, 37.9, 31.0) + 0.362 * g(nm, 442.0, 16.0, 26.7) - 0.065 * g(nm, 501.1, 20.4, 26.2);
    const y = 0.821 * g(nm, 568.8, 46.9, 40.5) + 0.286 * g(nm, 530.9, 16.3, 31.1);
    const z = 1.217 * g(nm, 437.0, 11.8, 36.0) + 0.681 * g(nm, 459.0, 26.0, 13.8);
    const rgb = [
      Math.max(0, 3.2406 * x - 1.5372 * y - 0.4986 * z),
      Math.max(0, -0.9689 * x + 1.8758 * y + 0.0415 * z),
      Math.max(0, 0.0557 * x - 0.204 * y + 1.057 * z),
    ];
    rgb.forEach((v, k) => (sum[k] += v));
    rows.push({ um: nm / 1000, rgb });
  }
  rows.forEach((r) => (r.rgb = r.rgb.map((v, k) => v / sum[k])));
  return rows;
}

// A bundle of ripples: random directions, log-spread wavelengths, gravity-capillary speeds.
function makeWaves(seed) {
  const rnd = mulberry32(seed);
  const K0 = 5.4; // rms surface curvature (1/m): strong enough that rays fold into a full web
  const raw = [];
  for (let i = 0; i < NW; i++) {
    const dir = ((i + 0.15 + 0.7 * rnd()) / NW) * TAU;
    const lambda = 0.035 * Math.pow(0.26 / 0.035, rnd());
    const k = TAU / lambda;
    raw.push({ dir, k, w: Math.pow(lambda / 0.12, -0.4), phi: rnd() * TAU, omega: Math.sqrt(9.81 * k + 7.3e-5 * k ** 3) });
  }
  const norm = Math.sqrt(raw.reduce((s, r) => s + r.w * r.w, 0) / 2);
  return raw.map((r) => ({ ...r, kx: Math.cos(r.dir) * r.k, ky: Math.sin(r.dir) * r.k, curv: (r.w / norm) * K0 }));
}

function hexToLinear(hex) {
  let h = hex.replace("#", "");
  if (h.length === 3) h = h.replace(/./g, "$&$&");
  const v = parseInt(h, 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255].map((c) => Math.pow(c / 255, 2.2));
}

/* ── renderer ────────────────────────────────────────────────────────────── */

function start(canvas, propsRef, q) {
  const gl = canvas.getContext("webgl2", {
    antialias: false, alpha: false, depth: false, stencil: false, powerPreference: "high-performance",
  });
  if (!gl) return () => {};
  const f32 = !!gl.getExtension("EXT_color_buffer_float");
  if (!f32 && !gl.getExtension("EXT_color_buffer_half_float")) return () => {};

  let dispose = init();
  const onLost = (e) => e.preventDefault();
  const onRestored = () => { dispose(); dispose = init(); };
  canvas.addEventListener("webglcontextlost", onLost);
  canvas.addEventListener("webglcontextrestored", onRestored);
  return () => {
    canvas.removeEventListener("webglcontextlost", onLost);
    canvas.removeEventListener("webglcontextrestored", onRestored);
    dispose();
  };

  function init() {
    const program = (vs, fs) => {
      const p = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s); gl.deleteShader(s);
      }
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      return p;
    };
    const uniforms = (p, names) => Object.fromEntries(names.map((n) => [n, gl.getUniformLocation(p, n)]));

    const pSurf = program(VS_FULL, FS_SURFACE);
    const pCaus = program(VS_CAUSTIC, FS_CAUSTIC);
    const pPost = program(VS_FULL, FS_POST);
    const uS = uniforms(pSurf, ["uWave", "uPhase", "uSize", "uGrid"]);
    const uC = uniforms(pCaus, ["uSurf", "uCells", "uSize", "uView", "uDepth", "uLight", "uIor", "uIor0", "uWeight", "uPixArea", "uMaxI"]);
    const uP = uniforms(pPost, ["uAccum", "uRes", "uFloor", "uLightC", "uGain", "uContrast", "uGrain", "uFrame"]);
    gl.useProgram(pCaus); gl.uniform1i(uC.uSurf, 0);
    gl.useProgram(pPost); gl.uniform1i(uP.uAccum, 0);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);

    const tex = (w, h, ifmt, type, filter) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, gl.RGBA, type, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    const fbo = (t) => {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      return f;
    };

    let spectrum = null, nSamples = 0;
    const weights = new Float32Array(MAX_S * 3);
    const setSamples = (n) => {
      if (n === nSamples) return;
      nSamples = n; spectrum = spectralSamples(n);
      weights.fill(0); spectrum.forEach((s, i) => weights.set(s.rgb, i * 3));
    };
    const iorArr = new Float32Array(MAX_S);
    const waveArr = new Float32Array(NW * 4);
    const phaseArr = new Float32Array(NW);

    let W = 0, H = 0, nx = 0, ny = 0, idxCount = 0, dpr = 1;
    let accumTex = null, accumFbo = null, surfTex = null, surfFbo = null;
    let waves = null, waveSeed = null;
    let sig = "", t = 0, last = performance.now(), raf = 0, visible = true, needsResize = true;

    const marginOf = (p) => 0.02 + p.depth * (0.08 * p.ripple + 0.0016 * p.dispersion);

    function resize() {
      needsResize = false;
      const p = propsRef.current;
      dpr = Math.min(window.devicePixelRatio || 1, q.dpr);
      const cw = Math.round(canvas.clientWidth * dpr), ch = Math.round(canvas.clientHeight * dpr);
      if (cw < 2 || ch < 2) return false;
      if (cw === W && ch === H) return true;
      W = cw; H = ch; canvas.width = W; canvas.height = H;

      const devPpm = PX_PER_METER * p.zoom * dpr;
      const m2 = 2 * marginOf(p) * devPpm;
      const cellPx = Math.max(q.cellPx, Math.sqrt(((W + m2) * (H + m2)) / 450000)); // cap ~450k cells
      nx = Math.max(8, Math.round((W + m2) / cellPx));
      ny = Math.max(8, Math.round((H + m2) / cellPx));

      [accumTex, surfTex].forEach((x) => x && gl.deleteTexture(x));
      [accumFbo, surfFbo].forEach((x) => x && gl.deleteFramebuffer(x));
      accumTex = tex(W, H, gl.RGBA16F, gl.HALF_FLOAT, gl.LINEAR);
      accumFbo = fbo(accumTex);
      surfTex = tex(nx + 1, ny + 1, f32 ? gl.RGBA32F : gl.RGBA16F, f32 ? gl.FLOAT : gl.HALF_FLOAT, gl.NEAREST);
      surfFbo = fbo(surfTex);

      const stride = nx + 1;
      const idx = new Uint32Array(nx * ny * 6);
      let o = 0;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const a = j * stride + i, b = a + 1, c = a + stride, d = c + 1;
        idx[o++] = a; idx[o++] = b; idx[o++] = c; idx[o++] = b; idx[o++] = d; idx[o++] = c;
      }
      gl.bindVertexArray(vao);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
      idxCount = idx.length;
      return true;
    }

    function draw(now) {
      const p = propsRef.current;
      if (!waves || waveSeed !== p.seed) { waves = makeWaves(p.seed); waveSeed = p.seed; }

      const Wm = canvas.clientWidth / (PX_PER_METER * p.zoom);
      const Hm = canvas.clientHeight / (PX_PER_METER * p.zoom);
      const m = marginOf(p);
      const size = [Wm + 2 * m, Hm + 2 * m];

      waves.forEach((w, i) => {
        waveArr.set([w.kx, w.ky, (p.ripple * w.curv) / (w.k * w.k), 0], i * 4);
        phaseArr[i] = (((w.phi - w.omega * p.speed * t) % TAU) + TAU) % TAU;
      });

      // enough wavelengths that neighbouring copies of a caustic line sit ≲2px apart
      const spreadPx = p.dispersion * p.depth * 0.0028 * PX_PER_METER * p.zoom * 1.6;
      setSamples(Math.min(MAX_S, Math.max(q.samples, Math.ceil(spreadPx / 2))));

      const n0 = waterIor(0.55);
      spectrum.forEach((s, i) => (iorArr[i] = n0 + p.dispersion * (waterIor(s.um) - n0)));
      const a = (p.lightAngle * Math.PI) / 180;
      const light = [Math.sin(a) * 0.8, Math.cos(a), Math.sin(a) * 0.6];

      gl.bindVertexArray(vao);
      gl.disable(gl.BLEND);

      // 1) water surface
      gl.bindFramebuffer(gl.FRAMEBUFFER, surfFbo);
      gl.viewport(0, 0, nx + 1, ny + 1);
      gl.useProgram(pSurf);
      gl.uniform4fv(uS.uWave, waveArr); gl.uniform1fv(uS.uPhase, phaseArr);
      gl.uniform2f(uS.uSize, size[0], size[1]); gl.uniform2f(uS.uGrid, nx, ny);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      // 2) refract + splat one mesh per wavelength, additively
      gl.bindFramebuffer(gl.FRAMEBUFFER, accumFbo);
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(pCaus);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, surfTex);
      gl.uniform2i(uC.uCells, nx, ny);
      gl.uniform2f(uC.uSize, size[0], size[1]); gl.uniform2f(uC.uView, Wm, Hm);
      gl.uniform1f(uC.uDepth, p.depth); gl.uniform3f(uC.uLight, ...light);
      gl.uniform1fv(uC.uIor, iorArr); gl.uniform1f(uC.uIor0, n0); gl.uniform3fv(uC.uWeight, weights);
      gl.uniform1f(uC.uPixArea, (Wm / W) * (Hm / H)); gl.uniform1f(uC.uMaxI, 14);
      gl.drawElementsInstanced(gl.TRIANGLES, idxCount, gl.UNSIGNED_INT, 0, spectrum.length);
      gl.disable(gl.BLEND);

      // 3) grade to screen
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W, H);
      gl.useProgram(pPost);
      gl.bindTexture(gl.TEXTURE_2D, accumTex);
      gl.uniform2f(uP.uRes, W, H);
      gl.uniform3fv(uP.uFloor, hexToLinear(p.floorColor));
      gl.uniform3fv(uP.uLightC, hexToLinear(p.lightColor));
      gl.uniform1f(uP.uGain, 0.46 * p.exposure); gl.uniform1f(uP.uContrast, 1.95);
      gl.uniform1f(uP.uGrain, p.grain);
      gl.uniform1f(uP.uFrame, (Math.floor(now / 42) % 997) * 17.31);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)");

    function tick(now) {
      raf = 0;
      if (!visible) return;
      const p = propsRef.current;
      if (needsResize && !resize() && !W) { raf = requestAnimationFrame(tick); return; }
      const animate = !p.paused && !(reduced && reduced.matches);
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (animate) t += dt;
      const s = animate ? "" : JSON.stringify(p) + W + "x" + H; // when frozen only redraw on changes
      if (animate || s !== sig) { sig = s; draw(now); }
      raf = requestAnimationFrame(tick);
    }
    const kick = () => { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } };

    t = 40; // start mid-motion so a frozen frame still looks good
    const ro = new ResizeObserver(() => { needsResize = true; sig = ""; kick(); });
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) kick(); });
    io.observe(canvas);
    kick();

    return () => {
      cancelAnimationFrame(raf); raf = 0; visible = false;
      ro.disconnect(); io.disconnect();
      [accumTex, surfTex].forEach((x) => x && gl.deleteTexture(x));
      [accumFbo, surfFbo].forEach((x) => x && gl.deleteFramebuffer(x));
      [pSurf, pCaus, pPost].forEach((x) => gl.deleteProgram(x));
      gl.deleteBuffer(ibo); gl.deleteVertexArray(vao);
    };
  }
}

/* ── component ───────────────────────────────────────────────────────────── */

export default function PoolCaustics({
  children,
  className,
  style,
  dispersion = 6, // 1 = real water · ~4 = slight rainbow · 20 = the video's "exaggerated"
  depth = 0.95, // metres to the floor: shallow = soft glow, ~1 = sharp, deeper = tangled
  speed = 0.05, // 1 = real ripple speed (frantic); ~0.2 is calm
  zoom = 1.618, // >1 zooms in
  ripple = 1, // wave strength
  exposure = 1,
  grain = 0.045,
  lightAngle = 21.7, // degrees from vertical (the angle used in the video)
  floorColor = "#008DB9",
  lightColor = "#00BBFF",
  seed = 7,
  quality = "medium", // "low" | "medium" | "high"
  paused = false,
  ...rest
}) {
  const canvasRef = useRef(null);
  const propsRef = useRef({});
  propsRef.current = { dispersion, depth, speed, zoom, ripple, exposure, grain, lightAngle, floorColor, lightColor, seed, paused };

  useEffect(() => start(canvasRef.current, propsRef, QUALITY[quality] || QUALITY.medium), [quality]);

  return (
    <div
      className={className}
      {...rest}
      style={{ position: "relative", overflow: "hidden", isolation: "isolate", background: floorColor, ...style }}
    >
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", zIndex: -1, pointerEvents: "none", display: "block" }}
      />
      {children}
    </div>
  );
}
