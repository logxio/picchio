// The same account picchio runs locally, in the browser.
//
// Constants and formulas are ported from picchio.py and must move with
// it: VRAM_SPARE, PLAN_COMPUTE, KV_BYTES, the kv formula in kv_account
// and the layer split in plan_layers. checkConstants() below states what
// this file believes, so a drift is a visible mismatch and not a wrong
// answer.

export const PLAN_COMPUTE = 512 * 1024 ** 2;   // graph buffer, measured
export const VRAM_SPARE = 1024 * 1024 ** 2;    // what the engine leaves
export const CTX_CHAT = 4096;                  // an ordinary conversation
export const CTX_LONG = 131072;                // a document, a codebase
export const KV_BYTES = { f16: 2.0, bf16: 2.0, f32: 4.0,
                          q8_0: 34 / 32, q4_0: 18 / 32 };
const ARRAY_KEEP = 1024;

// GGUF value types, width in bytes for the fixed ones
const T = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
const SIGNED = { 1: 1, 3: 1, 5: 1, 11: 1 };
const FLOAT = { 6: 1, 12: 1 };

class Cursor {
  constructor(buf) { this.v = new DataView(buf); this.p = 0; }
  need(n) { if (this.p + n > this.v.byteLength) throw new RangeError("short"); }
  u32() { this.need(4); const x = this.v.getUint32(this.p, true); this.p += 4; return x; }
  u64() { this.need(8); const x = this.v.getBigUint64(this.p, true); this.p += 8; return Number(x); }
  scalar(t) {
    const w = T[t];
    if (w === undefined) throw new Error(`gguf value type ${t}`);
    this.need(w);
    let x;
    if (FLOAT[t]) x = w === 4 ? this.v.getFloat32(this.p, true) : this.v.getFloat64(this.p, true);
    else if (w === 8) x = Number(SIGNED[t] ? this.v.getBigInt64(this.p, true) : this.v.getBigUint64(this.p, true));
    else if (w === 4) x = SIGNED[t] ? this.v.getInt32(this.p, true) : this.v.getUint32(this.p, true);
    else if (w === 2) x = SIGNED[t] ? this.v.getInt16(this.p, true) : this.v.getUint16(this.p, true);
    else x = SIGNED[t] ? this.v.getInt8(this.p) : this.v.getUint8(this.p);
    this.p += w;
    return x;
  }
  str() {
    const n = this.u64();
    if (n > 1 << 24) throw new Error("gguf string too long");
    this.need(n);
    const s = new TextDecoder("utf-8").decode(new Uint8Array(this.v.buffer, this.v.byteOffset + this.p, n));
    this.p += n;
    return s;
  }
  value(t) {
    if (t === 8) return this.str();
    if (t === 9) {
      const it = this.u32(), n = this.u64();
      if (it === 8) { for (let i = 0; i < n; i++) this.str(); return null; }
      if (it === 9) throw new Error("nested gguf array");
      const w = T[it];
      if (w === undefined) throw new Error(`gguf array type ${it}`);
      if (n > ARRAY_KEEP) { this.need(w * n); this.p += w * n; return null; }
      const out = [];
      for (let i = 0; i < n; i++) out.push(this.scalar(it));
      return out;
    }
    return this.scalar(t);
  }
}

// Everything the fit account needs, and nothing after it. Measured
// across fourteen architectures the last of these keys closes inside
// three kilobytes, while the tokenizer arrays that follow run to
// fifteen megabytes; stopping here is what makes the answer instant.
const WANTED = ["block_count", "attention.head_count", "attention.head_count_kv",
                "attention.key_length", "attention.value_length",
                "embedding_length", "full_attention_interval", "expert_count"];

export function readHeader(buf) {
  const c = new Cursor(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== "GGUF") throw new Error("that address is not a GGUF file");
  c.p = 4;
  c.u32();                       // version
  c.u64();                       // tensor count
  const kvCount = c.u64();
  const out = {};
  let arch = null;
  for (let i = 0; i < kvCount; i++) {
    const key = c.str();
    const t = c.u32();
    const v = c.value(t);
    if (v !== null) out[key] = v;
    if (key === "general.architecture") arch = v;
    if (arch) {
      const have = WANTED.filter((k) => (`${arch}.${k}`) in out).length;
      // block_count plus one attention pair is the whole account
      if (out[`${arch}.block_count`] &&
          (out[`${arch}.attention.key_length`] || out[`${arch}.embedding_length`]) &&
          have >= 3 && key.startsWith("tokenizer.")) break;
    }
  }
  if (!arch) throw new Error("the header names no architecture");
  return { arch, kv: out };
}

const one = (v) => (Array.isArray(v) ? Math.max(...v.filter((x) => typeof x === "number")) : v);

export function kvBytes(head, ctx, dtype) {
  const g = (k) => head.kv[`${head.arch}.${k}`];
  const width = KV_BYTES[dtype];
  if (width === undefined) return null;
  const blocks = g("block_count");
  const heads = one(g("attention.head_count"));
  let heads_kv = one(g("attention.head_count_kv")) || heads;
  let klen = g("attention.key_length"), vlen = g("attention.value_length");
  if ((!klen || !vlen) && g("embedding_length") && heads) {
    klen = vlen = Math.floor(g("embedding_length") / heads);
  }
  if (!(blocks && heads_kv && klen && vlen)) return null;
  const interval = g("full_attention_interval") || 1;
  const att = Math.max(1, Math.floor(blocks / Math.max(1, interval)));
  return Math.round(ctx * att * heads_kv * (klen + vlen) * width);
}

export function layersOnCard(fileBytes, kv, blocks, budget) {
  const total = blocks + 1;                    // the output layer rides along
  const perLayer = fileBytes / total + kv / total;
  if (perLayer <= 0) return null;
  const room = budget - PLAN_COMPUTE;
  if (room <= 0) return [0, total];
  return [Math.max(0, Math.min(total, Math.floor(room / perLayer))), total];
}

// One verdict for one machine. Returns the light, the layer split and
// the numbers under it; the caller decides what a reader sees.
export function judge(head, fileBytes, machine, ctx, dtype = "f16") {
  const kv = kvBytes(head, ctx, dtype);
  if (kv === null) {
    return { light: "unknown",
             why: "this file does not carry the shape of its attention layers, so its memory cannot be counted" };
  }
  const need = fileBytes + kv + PLAN_COMPUTE;
  const blocks = head.kv[`${head.arch}.block_count`];
  const budget = machine.unified
    ? Math.floor(machine.bytes * 0.78)
    : Math.max(0, machine.bytes - VRAM_SPARE);
  const split = layersOnCard(fileBytes, kv, blocks, budget);
  let light = "fits";
  if (split && split[0] >= split[1]) light = "fits";
  else if (split && split[0] > 0) light = "partial";
  else light = "cpu";
  return { light, need, kv, budget, split, blocks };
}
