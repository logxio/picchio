"""Read a model's shape without downloading the model.

The question people ask before a 20 GB download is whether it will fit,
and answering it by first spending the 20 GB is no answer at all. A GGUF
file carries its whole geometry in a header at byte zero, so a ranged GET
of the first few MiB is enough: the tensor data that follows is never
read. An ollama tag is one step further back, its manifest naming the
blob and its size before any byte of the blob is fetched.

Nothing here falls back to a full download, and nothing guesses a size:
a source that will not serve a range, or a header that will not parse
inside the cap, is reported as such.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "picchio"
OLLAMA_REGISTRY = "https://registry.ollama.ai"
FIRST_WINDOW = 1024 * 1024
MAX_WINDOW = 64 * 1024 * 1024


class RemoteError(Exception):
    """A remote source could not answer. The message is what the user
    sees, so it names the source and what it did."""


def _get(url, start=None, end=None, timeout=30):
    """One ranged GET. Returns (status, headers, body-or-stream)."""
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    if start is not None:
        request.add_header("Range", "bytes={}-{}".format(start, end))
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RemoteError("{} answered HTTP {}".format(
            urllib.parse.urlsplit(url).netloc, exc.code))
    except urllib.error.URLError as exc:
        raise RemoteError("{} did not answer: {}".format(
            urllib.parse.urlsplit(url).netloc, exc.reason))
    except OSError as exc:
        raise RemoteError("{} did not answer: {}".format(
            urllib.parse.urlsplit(url).netloc, exc))


def _total_size(response, fetched):
    """The whole file's size, from the range reply's own accounting.

    Content-Range is authoritative on a 206; Content-Length only tells
    the truth when the server ignored the range and sent everything."""
    rng = response.headers.get("Content-Range") or ""
    match = re.search(r"/(\d+)\s*$", rng)
    if match:
        return int(match.group(1))
    if response.status == 200:
        length = response.headers.get("Content-Length")
        if length and length.isdigit():
            return int(length)
    return None


def head_bytes(url, reader, window=FIRST_WINDOW, cap=MAX_WINDOW):
    """(parsed header, total file size). Fetches a prefix and grows it
    until the reader is satisfied or the cap is reached.

    Tokenizer vocabularies live in the same table as the geometry and
    can push the end of the header megabytes in, so the window doubles
    rather than betting on one size. Each attempt is a fresh ranged GET:
    the reader consumes a stream once and cannot be rewound."""
    last = None
    while window <= cap:
        response = _get(url, 0, window - 1)
        with response:
            body = response.read()
            total = _total_size(response, len(body))
        if not body:
            raise RemoteError("the source returned no bytes")
        if body[:4] != b"GGUF":
            raise RemoteError("that is not a GGUF file (no magic at the "
                              "start)")
        try:
            import io
            return reader(io.BytesIO(body)), total
        except Exception as exc:      # header runs past this prefix
            last = exc
            if len(body) < window:    # the whole file is here already
                raise RemoteError("the header did not parse even with the "
                                  "whole file: {}".format(exc))
            window *= 2
    raise RemoteError("the header did not parse within {} MiB ({})".format(
        cap // (1024 ** 2), last))


HF_HOSTS = ("huggingface.co", "hf-mirror.com", "www.huggingface.co")


def looks_remote(arg):
    return str(arg).startswith(("http://", "https://"))


def hf_resolve(arg):
    """A direct .gguf URL for a huggingface-shaped argument, or None.

    Accepts a full resolve/blob url, and the shorthand a model card
    shows: repo owner, repo name and file."""
    if looks_remote(arg):
        parts = urllib.parse.urlsplit(arg)
        if parts.netloc not in HF_HOSTS:
            return arg if arg.lower().endswith(".gguf") else None
        path = parts.path.replace("/blob/", "/resolve/")
        return urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, path, parts.query, ""))
    match = re.match(r"^([\w.-]+)/([\w.-]+)/([\w.@+-]+\.gguf)$", str(arg))
    if match:
        return "https://huggingface.co/{}/{}/resolve/main/{}".format(*
                                                                    match.groups())
    return None


def ollama_ref(tag):
    """(namespace/name, version) for a registry lookup, or None."""
    text = str(tag or "").strip()
    if not text or looks_remote(text) or text.lower().endswith(".gguf"):
        return None
    name, _, version = text.partition(":")
    if not re.match(r"^[\w.-]+(/[\w.-]+)?$", name):
        return None
    if "/" not in name:
        name = "library/" + name
    return name, (version or "latest")


def ollama_manifest(tag):
    """(model blob digest, blob size) straight from the registry.

    The manifest is a few hundred bytes and names the weights layer with
    its size, so the fit answer needs no blob at all when the header is
    not wanted."""
    ref = ollama_ref(tag)
    if not ref:
        raise RemoteError("{!r} is not an ollama tag".format(tag))
    name, version = ref
    url = "{}/v2/{}/manifests/{}".format(OLLAMA_REGISTRY, name, version)
    response = _get(url)
    with response:
        try:
            body = json.loads(response.read().decode("utf-8"))
        except ValueError as exc:
            raise RemoteError("the registry sent no readable manifest: "
                              "{}".format(exc))
    for layer in body.get("layers") or []:
        if str(layer.get("mediaType", "")).endswith(".model"):
            digest, size = layer.get("digest"), layer.get("size")
            if digest and size:
                return digest, int(size)
    raise RemoteError("the manifest names no model layer")


def ollama_blob_url(tag, digest):
    name, _version = ollama_ref(tag)
    return "{}/v2/{}/blobs/{}".format(OLLAMA_REGISTRY, name, digest)
