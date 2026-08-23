// A read-only window onto the Ollama registry for the browser.
//
// The registry serves no CORS headers, so a page cannot ask it what a
// tag weighs. This forwards exactly two questions and nothing else: the
// manifest for a tag, and a ranged read of the model blob's first bytes.
// It is not a general proxy; the host is fixed and both path shapes are
// validated before anything leaves.

const REGISTRY = "https://registry.ollama.ai";
const NAME = /^[a-zA-Z0-9][\w.-]*(\/[a-zA-Z0-9][\w.-]*)?$/;
const TAG = /^[\w.-]+$/;
const DIGEST = /^sha256:[a-f0-9]{64}$/;
const MAX_RANGE = 8 * 1024 * 1024;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Range",
  "Access-Control-Expose-Headers": "Content-Range, Content-Length",
};

function fail(status, message) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

function fullName(name) {
  return name.includes("/") ? name : `library/${name}`;
}

// A range the browser asked for, clamped so this never becomes a way to
// pull whole models through the worker.
function clampRange(header) {
  const match = /^bytes=(\d+)-(\d+)?$/.exec(header || "");
  if (!match) return `bytes=0-${MAX_RANGE - 1}`;
  const start = Number(match[1]);
  const end = match[2] === undefined ? start + MAX_RANGE - 1 : Number(match[2]);
  return `bytes=${start}-${Math.min(end, start + MAX_RANGE - 1)}`;
}

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS });
    }
    if (request.method !== "GET") return fail(405, "GET only");

    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean);

    if (parts[0] === "manifest") {
      const name = parts.slice(1, -1).join("/");
      const tag = parts[parts.length - 1];
      if (!NAME.test(name) || !TAG.test(tag)) return fail(400, "bad tag");
      const upstream = await fetch(
        `${REGISTRY}/v2/${fullName(name)}/manifests/${tag}`,
        { headers: { "User-Agent": "picchio-fit" } }
      );
      if (!upstream.ok) {
        return fail(upstream.status === 404 ? 404 : 502,
          upstream.status === 404 ? "no such model or tag" : "registry error");
      }
      return new Response(await upstream.text(), {
        headers: { ...CORS, "Content-Type": "application/json",
                   "Cache-Control": "public, max-age=3600" },
      });
    }

    if (parts[0] === "blob") {
      const digest = parts[parts.length - 1];
      const name = parts.slice(1, -1).join("/");
      if (!NAME.test(name) || !DIGEST.test(digest)) return fail(400, "bad blob");
      const upstream = await fetch(
        `${REGISTRY}/v2/${fullName(name)}/blobs/${digest}`,
        { headers: { Range: clampRange(request.headers.get("Range")),
                     "User-Agent": "picchio-fit" } }
      );
      if (!upstream.ok && upstream.status !== 206) {
        return fail(502, `registry answered ${upstream.status}`);
      }
      const headers = new Headers(CORS);
      for (const key of ["Content-Range", "Content-Length", "Content-Type"]) {
        const value = upstream.headers.get(key);
        if (value) headers.set(key, value);
      }
      headers.set("Cache-Control", "public, max-age=3600");
      return new Response(upstream.body, { status: upstream.status, headers });
    }

    return fail(404, "unknown path");
  },
};
