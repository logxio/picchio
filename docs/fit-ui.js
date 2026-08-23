import { judge, readHeader, CTX_CHAT, CTX_LONG } from "./fit.js";

// A tag needs the ollama registry, which serves no CORS headers, so it
// goes through a worker that forwards exactly two questions. A Hugging
// Face link needs nothing: that host answers the browser directly.
const WORKER = location.hostname === "localhost" || location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8787"        // wrangler dev, for working on this page
  : "https://fit.yansu.me";
const HF = ["huggingface.co", "hf-mirror.com"];
const WINDOW = 256 * 1024;   // every key the account needs closes in 3 KB
const GROW = [WINDOW, 4 << 20, 16 << 20, 64 << 20];

// Memory is the only thing the account reads off this list, so the
// labels exist to be recognised: somebody knows they bought a 3090
// without knowing it holds 24 GB. Every size a buyer can actually have
// is here, which is what the Mac side was missing most: the machines
// people buy to run a 70B are the 64, 96 and 128 GB ones.
const MACHINES = [
  ["Mac 8 GB", 8, true], ["Mac 16 GB", 16, true], ["Mac 24 GB", 24, true],
  ["Mac 32 GB", 32, true], ["Mac 36 GB", 36, true], ["Mac 48 GB", 48, true],
  ["Mac 64 GB", 64, true], ["Mac 96 GB", 96, true],
  ["Mac 128 GB", 128, true],
  ["RTX 3060 12 GB", 12, false], ["RTX 3080 10 GB", 10, false],
  ["RTX 3090 24 GB", 24, false],
  ["RTX 4060 8 GB", 8, false], ["RTX 4060 Ti 16 GB", 16, false],
  ["RTX 4070 12 GB", 12, false], ["RTX 4070 Ti S 16 GB", 16, false],
  ["RTX 4080 16 GB", 16, false], ["RTX 4090 24 GB", 24, false],
  ["RTX 5070 Ti 16 GB", 16, false], ["RTX 5080 16 GB", 16, false],
  ["RTX 5090 32 GB", 32, false],
  ["RX 7900 XTX 24 GB", 24, false], ["RX 7800 XT 16 GB", 16, false],
];

const $ = (id) => document.getElementById(id);
const gib = (n) => `${(n / 1024 ** 3).toFixed(1)} GiB`;
let machine = null;
let last = null;

function chips() {
  MACHINES.forEach(([label, gb, unified], i) => {
    const b = document.createElement("button");
    b.className = "m";
    b.textContent = label;
    b.setAttribute("aria-pressed", "false");
    b.onclick = () => {
      [...$("machines").children].forEach((c) => c.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      machine = { label, bytes: gb * 1024 ** 3, unified };
      run();
    };
    $("machines").appendChild(b);
    if (i === 1) b.click();
  });
}

function card(kind, word, say, after) {
  return `<div class="light ${kind}"><div class="word">${word}</div>
    <div class="say">${say}</div>${after ? `<div class="after">${after}</div>` : ""}</div>`;
}

function show(html) { $("out").innerHTML = html; }

function target(text) {
  const s = text.trim();
  if (!s) return null;
  if (/^https?:\/\//i.test(s)) {
    const u = new URL(s);
    if (!HF.includes(u.hostname)) throw new Error("Only Hugging Face links are read here. An Ollama tag also works.");
    return { kind: "url", url: s.replace("/blob/", "/resolve/") };
  }
  const m = s.match(/^([\w.-]+)\/([\w.-]+)\/([\w.@+-]+\.gguf)$/);
  if (m) return { kind: "url", url: `https://huggingface.co/${m[1]}/${m[2]}/resolve/main/${m[3]}` };
  if (/^[\w.-]+(\/[\w.-]+)?(:[\w.-]+)?$/.test(s)) {
    const [name, tag] = s.split(":");
    return { kind: "tag", name, tag: tag || "latest", raw: s };
  }
  throw new Error("That is neither a Hugging Face link nor an Ollama tag.");
}

// The total size comes from the same reply that carries the header, so
// nothing extra is fetched to learn how big the file is.
async function pull(url, bytes) {
  const r = await fetch(url, { headers: { Range: `bytes=0-${bytes - 1}` } });
  if (!r.ok && r.status !== 206) {
    if (r.status === 401 || r.status === 403) throw new Error("That file is gated. Open it in a browser, accept its terms, then use the tag or a local copy.");
    if (r.status === 404) throw new Error("Nothing is published at that address.");
    throw new Error(`The host answered ${r.status}.`);
  }
  const range = r.headers.get("Content-Range") || "";
  const total = Number((range.match(/\/(\d+)\s*$/) || [])[1]) ||
                Number(r.headers.get("Content-Length")) || null;
  const buf = await r.arrayBuffer();
  return { buf, total };
}

async function header(t) {
  let url = t.kind === "url" ? t.url : null;
  let total = null;
  if (t.kind === "tag") {
    const m = await fetch(`${WORKER}/manifest/${t.name}/${t.tag}`);
    if (!m.ok) throw new Error(m.status === 404
      ? "No model or tag by that name in the Ollama registry."
      : "The Ollama registry could not be reached.");
    const body = await m.json();
    const layer = (body.layers || []).find((l) => String(l.mediaType).endsWith(".model"));
    if (!layer) throw new Error("That tag names no model layer.");
    total = layer.size;
    url = `${WORKER}/blob/${t.name}/${layer.digest}`;
  }
  let err = null;
  for (const size of GROW) {
    const got = await pull(url, size);
    total = total || got.total;
    try {
      return { head: readHeader(got.buf), total };
    } catch (e) {
      err = e;
      if (got.buf.byteLength < size) break;   // that was the whole file
    }
  }
  throw new Error(`The header did not parse. ${err ? err.message : ""}`);
}

async function run() {
  const raw = $("q").value;
  if (!raw.trim() || !machine) { show(""); return; }
  let t;
  try { t = target(raw); } catch (e) { show(card("dim", "—", e.message)); return; }
  if (!t) { show(""); return; }

  const key = `${t.kind}:${t.url || t.raw}`;
  if (!last || last.key !== key) {
    show(card("dim", "reading", "Fetching the header. The weights are not downloaded."));
    try {
      last = { key, ...(await header(t)) };
    } catch (e) {
      last = null;
      show(card("dim", "—", e.message));
      return;
    }
  }
  render(last.head, last.total);
}

function render(head, total) {
  const chat = judge(head, total, machine, CTX_CHAT);
  if (chat.light === "unknown") { show(card("dim", "—", chat.why)); return; }
  const long = judge(head, total, machine, CTX_LONG);

  let kind = "go", word = "FITS", say, after;
  if (chat.light === "fits") {
    say = `The whole model fits in ${machine.label}.`;
    after = "";
    if (long.light !== "fits") {
      kind = "wait"; word = "PARTIAL";
      say = `Fits ${machine.label} for normal chat, but long documents push it onto the CPU.`;
      after = "Every extra token of context needs more memory. At a "
            + "document-sized context this model stops fitting.";
    }
  } else if (chat.light === "partial") {
    say = `${chat.split[0]} of its ${chat.split[1]} layers fit ${machine.label}. `
        + "The rest run on the CPU, which is much slower.";
    kind = "wait"; word = "PARTIAL";
    after = "It will load and answer. The layers on the CPU set the pace.";
  } else {
    kind = "stop"; word = "CPU";
    say = `No layer fits ${machine.label}. The whole model runs on the CPU.`;
    after = "It will still answer, slowly enough that most people stop waiting.";
  }

  // When the two contexts answer differently, the single-context table
  // contradicts the card above it: a reader sees 5.9 against a 9.0
  // budget and every layer on the card, and concludes the calculator is
  // broken. The number that actually busts the budget is the long one,
  // so on that card it is the number on the page.
  const split = (r) => (r.split ? `${r.split[0]} of ${r.split[1]}` : "—");
  const over = (r) => (r.need > r.budget
    ? ` <span class="over">over budget</span>` : "");
  const parts = chat.light !== long.light
    ? [["memory it needs, normal chat", gib(chat.need) + over(chat)],
       ["memory it needs, long document", gib(long.need) + over(long)]]
    : [["memory it needs", gib(chat.need) + over(chat)]];
  const layers = chat.light !== long.light
    ? [["layers on the GPU, normal chat", split(chat)],
       ["layers on the GPU, long document", split(long)]]
    : [["layers on the GPU", split(chat)]];
  const rows = [
    ["weights", gib(total)],
    ...parts,
    [`${machine.label} budget`, gib(chat.budget)],
    ...layers,
  ];
  show(card(kind, word, say, after)
    + `<table>${rows.map(([k, v]) => `<tr><td class="k">${k}</td><td>${v}</td></tr>`).join("")}</table>`
    + `<div class="hook">Not sure if the engine is lying to you? Engines
         often report 100% GPU usage while your GPU sits at 0%.</div>`
    + `<a class="star" href="https://github.com/logxio/picchio#run-it">Verify your GPU</a>`);
}

chips();
let timer;
$("q").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 400); });
