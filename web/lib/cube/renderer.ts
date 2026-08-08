/**
 * renderer.ts — the analytics page's cube, sharing the landing page's look.
 *
 * WHAT THIS IS AND IS NOT A COPY OF
 * --------------------------------
 * The mesh (rounded cubies), the material shader (sticker mask, bevel,
 * inter-cubie AO, three softbox speculars) and the palette are lifted from
 * `public/landing.html` deliberately — the cube should read as the same
 * object in both places, and re-deriving the material would guarantee it
 * did not.
 *
 * What is NOT carried over is everything the landing page needs and this
 * does not: the scroll choreography, the roaming detector HUD, the motion-
 * energy accumulation pass and the bloom chain. Those exist to dramatise
 * the CV pipeline. Here the cube is an instrument, so it renders in one
 * pass onto a transparent canvas and the page's own background shows
 * through — which is also what makes it work in both light and dark theme
 * without a second palette.
 *
 * HOW IT VISUALISES A METRIC
 * --------------------------
 * Two knobs, both fed straight from `coach/` output:
 *
 *   tps       turns per second. The cube executes a real algorithm at
 *             exactly that rate, so "4.7 TPS" stops being a number and
 *             becomes something you can watch and compare.
 *   faceGain  per-face brightness. Face usage share maps to how lit each
 *             face is, so an over-used R face is literally the bright one.
 *
 * Face gain scales albedo only, never the speculars — a dimmed face keeps
 * its highlights and still reads as a glossy sticker rather than a hole.
 */

export type FaceKey = "U" | "D" | "L" | "R" | "F" | "B";

export interface Move {
  /** 0 = x, 1 = y, 2 = z */
  axis: 0 | 1 | 2;
  /** which layer along that axis */
  coord: -1 | 0 | 1;
  /** +1 / -1 quarter turn */
  dir: 1 | -1;
}

export interface CubeParams {
  /** Quarter turns per second. 0 holds the cube still. */
  tps: number;
  /** 0..1 albedo multiplier per face. */
  faceGain: Record<FaceKey, number>;
  /** The algorithm to execute, looped. */
  moves: Move[];
  /** Degrees per second of idle yaw. 0 holds the default orientation. */
  spin: number;
  paused: boolean;
  /**
   * Let the viewer drag the cube around. On release it eases back to the
   * default orientation rather than staying where it was left — face
   * identity here is position-relative (the "up" face is whichever one is
   * up), so a cube abandoned at an arbitrary angle stops being readable as
   * U/R/F at all.
   */
  interactive: boolean;
}

/* Shader face order is the mesh's: +x -x +y -y +z -z. */
const FACE_ORDER: FaceKey[] = ["R", "L", "U", "D", "F", "B"];

/** WCA quarter turns in this renderer's y-up, +z-front world. R and U are
 *  taken from landing.html's MOVES table, the rest by the same handedness. */
export const WCA: Record<string, Move> = {
  R: { axis: 0, coord: 1, dir: 1 },
  "R'": { axis: 0, coord: 1, dir: -1 },
  L: { axis: 0, coord: -1, dir: -1 },
  "L'": { axis: 0, coord: -1, dir: 1 },
  U: { axis: 1, coord: 1, dir: 1 },
  "U'": { axis: 1, coord: 1, dir: -1 },
  D: { axis: 1, coord: -1, dir: -1 },
  "D'": { axis: 1, coord: -1, dir: 1 },
  F: { axis: 2, coord: 1, dir: 1 },
  "F'": { axis: 2, coord: 1, dir: -1 },
  B: { axis: 2, coord: -1, dir: -1 },
  "B'": { axis: 2, coord: -1, dir: 1 },
};

export function parseAlg(alg: string): Move[] {
  return alg
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => WCA[w])
    .filter(Boolean);
}

const CUBE_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUV;
layout(location=3) in float aFace;
uniform mat4 uPV, uModel;
uniform mat3 uNrmM;
out vec3 vN; out vec3 vWP; out vec2 vUV; out vec3 vLocal;
flat out int vF;
void main(){
  vLocal = aPos;
  vec4 wp = uModel * vec4(aPos, 1.0);
  vWP = wp.xyz;
  vN = normalize(uNrmM * aNrm);
  vUV = aUV;
  vF = int(aFace + 0.5);
  gl_Position = uPV * wp;
}`;

const CUBE_FS = `#version 300 es
precision highp float;
in vec3 vN; in vec3 vWP; in vec2 vUV; in vec3 vLocal;
flat in int vF;
uniform vec3 uCols[6];
uniform int uSticker;
uniform int uNbr;
uniform vec3 uEye;
out vec4 frag;

const vec3 DIRS[6] = vec3[6](
  vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1));

float hash(vec2 p){ p = fract(p*vec2(234.34, 435.345)); p += dot(p, p+34.23); return fract(p.x*p.y); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
  return mix(mix(hash(i), hash(i+vec2(1,0)), f.x),
             mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y);
}
float sdRR(vec2 p, vec2 b, float r){
  vec2 d = abs(p) - b + r;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}
float softbox(vec3 R, vec3 C, vec3 U, float sx, float sy, float soft){
  float cd = dot(R, C);
  if (cd <= 0.02) return 0.0;
  vec3 Xa = normalize(cross(U, C));
  vec3 Ya = normalize(cross(C, Xa));
  vec2 a = vec2(dot(R, Xa), dot(R, Ya)) / cd;
  vec2 q = abs(a) - vec2(sx, sy);
  float d = max(q.x, q.y);
  return smoothstep(soft, -soft, d) * smoothstep(0.0, 0.25, cd);
}

void main(){
  vec2 p = vUV - 0.5;
  float d = sdRR(p, vec2(0.425), 0.105);
  float aa = fwidth(d) * 1.2;
  bool hasSticker = ((uSticker >> vF) & 1) == 1;
  float mask = hasSticker ? smoothstep(aa, -aa, d) : 0.0;

  vec3 albP = vec3(0.012, 0.012, 0.014);
  vec3 alb = mix(albP, uCols[vF], mask);
  float rough = mix(0.34, 0.155, mask);

  float fp  = vnoise(vUV*9.0 + float(vF)*7.31);
  float scr = vnoise(vec2(vUV.x*150.0 + float(vF)*13.7, vUV.y*7.0));
  rough += (fp - 0.5) * 0.055 + smoothstep(0.78, 0.96, scr) * 0.07;
  rough = clamp(rough, 0.05, 0.9);

  vec3 N = vN;
  if (hasSticker) {
    float e = 0.004;
    vec2 g = vec2(sdRR(p+vec2(e,0.0), vec2(0.425), 0.105) - d,
                  sdRR(p+vec2(0.0,e), vec2(0.425), 0.105) - d) / e;
    float bev = smoothstep(-0.05, -0.008, d) * mask;
    vec3 dpx = dFdx(vWP), dpy = dFdy(vWP);
    vec2 dux = dFdx(vUV), duy = dFdy(vUV);
    vec3 T = dpx*duy.y - dpy*dux.y;
    if (dot(T, T) > 1e-12) {
      T = normalize(T);
      vec3 B = normalize(cross(vN, T));
      N = normalize(vN + (T*g.x + B*g.y) * bev * 0.38);
    }
  }

  float ao = 1.0;
  for (int i = 0; i < 6; i++) {
    if (((uNbr >> i) & 1) == 1) {
      float dist = 0.5 - dot(vLocal, DIRS[i]);
      ao *= mix(0.42, 1.0, smoothstep(0.0, 0.46, dist));
    }
  }

  vec3 V = normalize(uEye - vWP);
  float ndv = clamp(dot(N, V), 0.0, 1.0);
  vec3 R = reflect(-V, N);

  vec3 ambDn = vec3(0.016, 0.017, 0.021);
  vec3 ambUp = vec3(0.075, 0.077, 0.086);
  vec3 amb = mix(ambDn, ambUp, N.y*0.5 + 0.5);

  vec3 keyDir = normalize(vec3(-0.5, 0.62, 0.6));
  float wrap = clamp((dot(N, keyDir) + 0.35) / 1.35, 0.0, 1.0);
  vec3 fillDir = normalize(vec3(0.7, 0.1, 0.4));
  float fill = clamp(dot(N, fillDir), 0.0, 1.0);

  vec3 diffuse = alb * (amb*1.9 + vec3(0.92,0.95,1.05)*wrap*0.52 + vec3(1.0,0.96,0.9)*fill*0.10);

  float soft = mix(0.05, 0.85, rough*rough);
  float energy = mix(1.0, 0.28, rough);
  float F = 0.045 + 0.955 * pow(1.0 - ndv, 5.0);

  vec3 spec =
      softbox(R, normalize(vec3( 0.15, 1.0,  0.38)), vec3(0,0,1), 1.7, 0.55, soft) * vec3(1.00, 0.99, 0.97) * 4.6
    + softbox(R, normalize(vec3(-0.72, 0.34, 0.58)), vec3(0,1,0), 0.34, 0.95, soft) * vec3(0.86, 0.93, 1.06) * 3.4
    + softbox(R, normalize(vec3( 0.80, 0.22,-0.46)), vec3(0,1,0), 0.13, 1.10, soft) * vec3(1.00, 0.97, 0.90) * 2.8;

  vec3 col = diffuse * ao + spec * F * energy * mix(0.6, 1.0, ao);
  /* Tone map + inverse gamma. The landing page does this in its post
     chain; with no post pass here it has to happen at the end of the
     material shader or everything above 1.0 clips to white. */
  col = col / (col + vec3(0.72));
  frag = vec4(pow(max(col, 0.0), vec3(1.0/2.2)), 1.0);
}`;

const HALF = 0.5;
const RAD = 0.088;
const SEG = 10;
const SPACING = 1.035;
const FOV = (32 * Math.PI) / 180;

const srgb2lin = (hex: string): number[] => {
  const v = parseInt(hex.slice(1), 16);
  return [16, 8, 0].map((sh) => Math.pow(((v >> sh) & 255) / 255, 2.2));
};
/* +x -x +y -y +z -z → right red, left orange, up white, down yellow,
   front green, back blue. Same palette as landing.html. */
const COLS = ["#D42A3D", "#FF5F0F", "#F4F6F9", "#FFD31A", "#00A067", "#0A5AC4"].map(
  srgb2lin,
);

type M3 = number[];
const I3: M3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function rot3(axis: number, deg: number): M3 {
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  if (axis === 0) return [1, 0, 0, 0, c, s, 0, -s, c];
  if (axis === 1) return [c, 0, -s, 0, 1, 0, s, 0, c];
  return [c, s, 0, -s, c, 0, 0, 0, 1];
}
function mul3(A: M3, B: M3): M3 {
  const C = new Array(9);
  for (let col = 0; col < 3; col++)
    for (let row = 0; row < 3; row++)
      C[col * 3 + row] =
        A[row] * B[col * 3] + A[3 + row] * B[col * 3 + 1] + A[6 + row] * B[col * 3 + 2];
  return C;
}
function mulV3(A: M3, v: number[]): number[] {
  return [
    A[0] * v[0] + A[3] * v[1] + A[6] * v[2],
    A[1] * v[0] + A[4] * v[1] + A[7] * v[2],
    A[2] * v[0] + A[5] * v[1] + A[8] * v[2],
  ];
}
function mul4(A: ArrayLike<number>, B: ArrayLike<number>): Float32Array {
  const C = new Float32Array(16);
  for (let col = 0; col < 4; col++)
    for (let row = 0; row < 4; row++)
      C[col * 4 + row] =
        A[row] * B[col * 4] +
        A[4 + row] * B[col * 4 + 1] +
        A[8 + row] * B[col * 4 + 2] +
        A[12 + row] * B[col * 4 + 3];
  return C;
}
function persp(fovY: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}
const easeIO = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/** Thrown when the GPU context died under us, as opposed to our code being
 *  wrong. The two need different responses — one is retryable, the other
 *  never will be — and they are indistinguishable from the return values
 *  alone, which is what made this bug so confusing. */
export class ContextLostError extends Error {
  constructor() {
    super("WebGL context lost");
    this.name = "ContextLostError";
  }
}

function compile(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const mk = (type: number, src: string) => {
    // A LOST CONTEXT LOOKS EXACTLY LIKE A BROKEN SHADER, and this is the
    // trap that produced a page-killing "shader compile failed" with no
    // detail. On a lost context every query returns null:
    // getShaderParameter(COMPILE_STATUS) is null (falsy, so it reads as
    // "did not compile") and getShaderInfoLog is null (so there is nothing
    // to print). Chrome allows 16 live WebGL contexts and force-loses the
    // oldest on the 17th — measured, not assumed — which a dev server doing
    // Fast Refresh remounts, or a few tabs, reaches easily. Check first.
    if (gl.isContextLost()) throw new ContextLostError();
    const sh = gl.createShader(type);
    if (!sh) throw new ContextLostError();
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      if (gl.isContextLost()) throw new ContextLostError();
      // The driver's log is the ONLY thing that identifies which line failed,
      // and GPUs disagree about what they accept — so it is quoted verbatim,
      // with the offending line pulled out beside it. An earlier version of
      // this threw a bare "shader compile failed", which said nothing and
      // took the whole page down with it.
      const log = gl.getShaderInfoLog(sh) || "(driver returned no log)";
      const line = Number(/ERROR:\s*\d+:(\d+)/.exec(log)?.[1] ?? 0);
      const src_line = line ? `\n  at line ${line}: ${src.split("\n")[line - 1]?.trim()}` : "";
      gl.deleteShader(sh);
      throw new Error(
        `${type === gl.VERTEX_SHADER ? "vertex" : "fragment"} shader: ${log}${src_line}`,
      );
    }
    return sh;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p) ?? "link failed");
  const u: Record<string, WebGLUniformLocation> = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(p, i)!;
    const name = info.name.replace("[0]", "");
    u[name] = gl.getUniformLocation(p, info.name)!;
  }
  return { p, u };
}

export interface CubeHandle {
  update(p: Partial<CubeParams>): void;
  destroy(): void;
}

export function createCubeRenderer(
  canvas: HTMLCanvasElement,
  initial: CubeParams,
): CubeHandle | null {
  const gl = canvas.getContext("webgl2", {
    antialias: true,
    alpha: true,
    premultipliedAlpha: false,
  });
  if (!gl) return null;

  /**
   * Hand the context back on any failure path.
   *
   * THIS IS THE BUG THAT MADE THE ORIGINAL FAILURE PERMANENT. A context is
   * allocated by getContext above, and the browser caps them at 16 per tab.
   * When setup failed after that point the function threw, React therefore
   * never ran the effect's cleanup (an effect that throws is not considered
   * mounted), and the context was leaked with nothing left holding a
   * reference to release it. Every subsequent remount — and a dev server
   * doing Fast Refresh remounts on each save — leaked one more, so a single
   * transient failure walked the tab to the cap and made the error stick.
   */
  const release = () => gl.getExtension("WEBGL_lose_context")?.loseContext();

  if (gl.isContextLost()) {
    release();
    throw new ContextLostError();
  }

  // Compilation depends on the viewer's GPU driver as well as on this code,
  // so a failure here is reported and swallowed rather than allowed to
  // unmount the route: the cube illustrates numbers that are all printed
  // beside it, so losing it must cost the illustration and never the page.
  // ContextLostError is re-thrown instead, because the caller can retry it.
  let prog: ReturnType<typeof compile>;
  try {
    prog = compile(gl, CUBE_VS, CUBE_FS);
  } catch (err) {
    release();
    if (err instanceof ContextLostError) throw err;
    console.error("[CubeViz] shader rejected by this driver —", err);
    return null;
  }

  /* ---- rounded-cubie mesh ---- */
  const faces = [
    { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
  ];
  const verts: number[] = [];
  const idx: number[] = [];
  const inner = HALF - RAD;
  faces.forEach((f, fi) => {
    const base = verts.length / 9;
    for (let j = 0; j <= SEG; j++)
      for (let i = 0; i <= SEG; i++) {
        const s = (i / SEG - 0.5) * 2 * HALF;
        const t = (j / SEG - 0.5) * 2 * HALF;
        const px = f.u[0] * s + f.v[0] * t + f.n[0] * HALF;
        const py = f.u[1] * s + f.v[1] * t + f.n[1] * HALF;
        const pz = f.u[2] * s + f.v[2] * t + f.n[2] * HALF;
        const qx = Math.max(-inner, Math.min(inner, px));
        const qy = Math.max(-inner, Math.min(inner, py));
        const qz = Math.max(-inner, Math.min(inner, pz));
        let nx = px - qx,
          ny = py - qy,
          nz = pz - qz;
        const nl = Math.hypot(nx, ny, nz) || 1;
        nx /= nl;
        ny /= nl;
        nz /= nl;
        verts.push(qx + nx * RAD, qy + ny * RAD, qz + nz * RAD, nx, ny, nz, i / SEG, j / SEG, fi);
      }
    for (let j = 0; j < SEG; j++)
      for (let i = 0; i < SEG; i++) {
        const a = base + j * (SEG + 1) + i,
          b = a + 1,
          c = a + SEG + 1,
          dd = c + 1;
        idx.push(a, b, c, b, dd, c);
      }
  });
  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
  const ibo = gl.createBuffer()!;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(idx), gl.STATIC_DRAW);
  const STRIDE = 36;
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, STRIDE, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, STRIDE, 12);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, STRIDE, 24);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 1, gl.FLOAT, false, STRIDE, 32);
  gl.bindVertexArray(null);
  const meshCount = idx.length;

  /* ---- cubies ---- */
  const DIRV = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];
  const cubies: { g: number[]; sticker: number; nbr: number }[] = [];
  for (let x = -1; x <= 1; x++)
    for (let y = -1; y <= 1; y++)
      for (let z = -1; z <= 1; z++) {
        if (!x && !y && !z) continue;
        let sticker = 0,
          nbr = 0;
        const g = [x, y, z];
        DIRV.forEach((d, f) => {
          const axis = d[0] ? 0 : d[1] ? 1 : 2;
          if (g[axis] === d[axis]) sticker |= 1 << f;
          else nbr |= 1 << f;
        });
        cubies.push({ g, sticker, nbr });
      }

  /** Persistent permutation state — mutated as whole moves complete, so a
   *  long-running animation costs nothing per frame beyond the in-flight
   *  layer. */
  let state = cubies.map((c) => ({ c, R: I3, pos: [c.g[0], c.g[1], c.g[2]] }));
  let applied = 0;

  let params: CubeParams = { ...initial };
  let W = 1,
    H = 1,
    dist = 10;
  const reduced =
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-reduced-motion: reduce)").matches;

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 1.75);
    W = Math.max(2, Math.round(canvas.clientWidth * dpr));
    H = Math.max(2, Math.round(canvas.clientHeight * dpr));
    canvas.width = W;
    canvas.height = H;
    const vmin = Math.min(canvas.clientWidth, canvas.clientHeight);
    /* 0.52, not the landing page's 0.40 (it shares the viewport with copy)
       and not higher: a cube yawing through 360 degrees presents its body
       diagonal at the corners, which is ~1.35x its face width, so anything
       much above this clips on the turn rather than on load — a bug that
       only appears seconds in. */
    const frac = (0.52 * vmin) / canvas.clientHeight;
    dist = (3 * SPACING + 0.12) / (2 * Math.tan(FOV / 2) * frac);
  }
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  const colsFlat = new Float32Array(18);
  const nrm = new Float32Array(9);
  let raf = 0;
  let t0 = 0;
  let moveClock = 0;
  /* Negative, so the camera sees +x and +z: the standard cubing view of
     U on top with F and R facing you. At +28 it showed L and F instead,
     which hid the second-most-used face on a panel whose whole subject is
     which faces get used. */
  let yaw = -28;

  /* ---- drag to look around ------------------------------------------- */
  const BASE_PITCH = 17;
  let dYaw = 0,
    dPitch = 0,
    dragging = false,
    lastX = 0,
    lastY = 0;

  const onDown = (e: PointerEvent) => {
    if (!params.interactive) return;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  };
  const onMove = (e: PointerEvent) => {
    if (!dragging) return;
    dYaw += (e.clientX - lastX) * 0.55;
    // Pitch is clamped: past vertical the cube reads as upside-down and the
    // face labels beside it stop matching what is on screen.
    dPitch = Math.max(-55, Math.min(55, dPitch + (e.clientY - lastY) * 0.4));
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onUp = (e: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      /* pointer already gone */
    }
    canvas.style.cursor = params.interactive ? "grab" : "";
  };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("pointerleave", onUp);
  if (initial.interactive) canvas.style.cursor = "grab";

  function frame(now: number) {
    // Stop cleanly if the context dies mid-flight rather than issuing GL
    // calls into a dead context every frame for the life of the page.
    if (gl!.isContextLost()) return;
    raf = requestAnimationFrame(frame);
    if (!t0) t0 = now;
    const dt = Math.min((now - t0) / 1000, 0.1);
    t0 = now;

    /* ---- advance the algorithm ---- */
    const moves = params.moves;
    if (!params.paused && params.tps > 0 && moves.length) {
      moveClock += dt * params.tps;
      const whole = Math.floor(moveClock);
      while (applied < whole) {
        const m = moves[applied % moves.length];
        const Rm = rot3(m.axis, m.dir * 90);
        for (const s of state) {
          if (Math.round(s.pos[m.axis]) !== m.coord) continue;
          s.pos = mulV3(Rm, s.pos);
          s.R = mul3(Rm, s.R);
        }
        applied++;
      }
    }
    if (!params.paused) yaw += dt * params.spin;

    // Ease back to the default orientation once the drag ends. Exponential,
    // framerate-corrected so it settles in the same wall-clock time on a
    // 60Hz and a 144Hz display rather than 2.4x faster on the latter.
    if (!dragging && (dYaw !== 0 || dPitch !== 0)) {
      const k = 1 - Math.pow(0.0005, dt);
      dYaw += -dYaw * k;
      dPitch += -dPitch * k;
      if (Math.abs(dYaw) < 0.02) dYaw = 0;
      if (Math.abs(dPitch) < 0.02) dPitch = 0;
    }

    /* ---- camera ---- */
    const proj = persp(FOV, W / H, dist - 6, dist + 6);
    const view = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -dist, 1]);
    const PV = mul4(proj, view);
    const Rg = mul3(rot3(0, BASE_PITCH + dPitch), rot3(1, yaw + dYaw));
    // prettier-ignore
    const group = new Float32Array([
      Rg[0], Rg[1], Rg[2], 0,
      Rg[3], Rg[4], Rg[5], 0,
      Rg[6], Rg[7], Rg[8], 0,
      0, 0, 0, 1,
    ]);

    /* ---- in-flight move ---- */
    const partial = moves.length && params.tps > 0 && !params.paused ? moveClock - applied : 0;
    const inFlight = partial > 0 ? moves[applied % moves.length] : null;
    const Rp = inFlight ? rot3(inFlight.axis, inFlight.dir * 90 * easeIO(Math.min(partial, 1))) : null;

    gl!.viewport(0, 0, W, H);
    gl!.clearColor(0, 0, 0, 0);
    gl!.clear(gl!.COLOR_BUFFER_BIT | gl!.DEPTH_BUFFER_BIT);
    gl!.enable(gl!.DEPTH_TEST);
    gl!.useProgram(prog.p);
    gl!.bindVertexArray(vao);

    for (let i = 0; i < 6; i++) {
      const gain = params.faceGain[FACE_ORDER[i]] ?? 1;
      colsFlat[i * 3] = COLS[i][0] * gain;
      colsFlat[i * 3 + 1] = COLS[i][1] * gain;
      colsFlat[i * 3 + 2] = COLS[i][2] * gain;
    }
    gl!.uniformMatrix4fv(prog.u.uPV, false, PV);
    gl!.uniform3fv(prog.u.uCols, colsFlat);
    gl!.uniform3f(prog.u.uEye, 0, 0, dist);

    for (const s of state) {
      let R = s.R;
      let pos = s.pos;
      if (Rp && inFlight && Math.round(s.pos[inFlight.axis]) === inFlight.coord) {
        R = mul3(Rp, s.R);
        pos = mulV3(Rp, s.pos);
      }
      // prettier-ignore
      const local = new Float32Array([
        R[0], R[1], R[2], 0,
        R[3], R[4], R[5], 0,
        R[6], R[7], R[8], 0,
        pos[0] * SPACING, pos[1] * SPACING, pos[2] * SPACING, 1,
      ]);
      const model = mul4(group, local);
      for (let cix = 0; cix < 3; cix++) {
        const c0 = model[cix * 4],
          c1 = model[cix * 4 + 1],
          c2 = model[cix * 4 + 2];
        const l = Math.hypot(c0, c1, c2) || 1;
        nrm[cix * 3] = c0 / l;
        nrm[cix * 3 + 1] = c1 / l;
        nrm[cix * 3 + 2] = c2 / l;
      }
      gl!.uniformMatrix4fv(prog.u.uModel, false, model);
      gl!.uniformMatrix3fv(prog.u.uNrmM, false, nrm);
      gl!.uniform1i(prog.u.uSticker, s.c.sticker);
      gl!.uniform1i(prog.u.uNbr, s.c.nbr);
      gl!.drawElements(gl!.TRIANGLES, meshCount, gl!.UNSIGNED_INT, 0);
    }
  }

  if (reduced) params.spin = 0;
  raf = requestAnimationFrame(frame);

  return {
    update(p) {
      const movesChanged = p.moves && p.moves !== params.moves;
      params = { ...params, ...p };
      if (reduced) params.spin = 0;
      canvas.style.cursor = params.interactive ? (dragging ? "grabbing" : "grab") : "";
      if (movesChanged) {
        /* A new algorithm restarts from solved: continuing a permutation
           built by a different move list would show a scrambled cube with
           no explanation. */
        state = cubies.map((c) => ({ c, R: I3, pos: [c.g[0], c.g[1], c.g[2]] }));
        applied = 0;
        moveClock = 0;
      }
    },
    destroy() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onUp);
      // Explicitly hand the context slot back instead of waiting for GC.
      // With a hard cap of 16 per browser and two cubes per page, a dev
      // server remounting on every save would otherwise walk straight into
      // the cap — which is precisely the bug this file now guards against.
      // Callers must ignore the `webglcontextlost` event this fires.
      gl!.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
}
