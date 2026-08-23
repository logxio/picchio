"""Read a model's shape without downloading the model.

The question people ask before a 20 GB download is whether it will fit,
and answering it by first spending the 20 GB is no answer at all. A GGUF
file carries its whole geometry in a header at byte zero, so a ranged GET
of the first few MiB is enough: the tensor data that follows is never
read. An ollama tag is one step further back, its manifest naming the
blob and its size before any byte of the blob is fetched.

Nothing here falls back to a full download and nothing guesses a size.
Every read is capped, including from a server that ignores the range and
starts sending the whole file, and a header that will not parse inside
the cap is reported instead of estimated around.
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


_GATED = (". The file is gated: open it in a browser, accept its terms, "
          "and download it yourself")
_WHAT_THE_CODE_MEANS = {
    401: _GATED, 403: _GATED,
    404: ". Nothing is published at that address, so check the spelling",
}


def _get(url, start=None, end=None, timeout=30):
    """One ranged GET, as an open response. Redirects are followed
    with the range intact, which is the whole path on hugging face:
    /resolve/ answers 302 and the file itself comes from a cdn host."""
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    if start is not None:
        request.add_header("Range", "bytes={}-{}".format(start, end))
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RemoteError("{} answered HTTP {}{}".format(
            urllib.parse.urlsplit(url).netloc, exc.code,
            _WHAT_THE_CODE_MEANS.get(exc.code, "")))
    except urllib.error.URLError as exc:
        raise RemoteError("{} did not answer: {}".format(
            urllib.parse.urlsplit(url).netloc, exc.reason))
    except OSError as exc:
        raise RemoteError("{} did not answer: {}".format(
            urllib.parse.urlsplit(url).netloc, exc))


def _total_size(response):
    """The whole file's size, from the reply's own accounting.

    Content-Range is authoritative on a 206. A 200 means the range was
    ignored and the body is the whole file, so its Content-Length is
    the total; on a 206 that same header measures only the slice."""
    rng = response.headers.get("Content-Range") or ""
    match = re.search(r"/(\d+)\s*$", rng)
    if match:
        return int(match.group(1))
    if getattr(response, "status", None) == 200:
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
    the reader consumes a stream once and cannot be rewound.

    The read is capped at the window whether or not the server honors
    the range. A source that answers 200 and starts sending 20 GB is
    hung up on after the window, so the promise in this module's first
    paragraph holds against servers that ignore Range as well."""
    last = None
    while window <= cap:
        response = _get(url, 0, window - 1)
        with response:
            body = response.read(window)
            total = _total_size(response)
        if not body:
            raise RemoteError("the source returned no bytes")
        if body[:4] != b"GGUF":
            raise RemoteError("that url does not serve a GGUF file (the "
                              "first bytes are not the GGUF magic)")
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


def model_url(arg):
    """The url to read the header from, or None when the argument names
    no url at all.

    A hugging face page url is the one shape that needs translating:
    the address in the browser bar says /blob/, which serves the html
    page around the file, and /resolve/ serves the file itself. Every
    other url is taken at its word rather than screened by its file
    extension. A link that carries the name in a query string is still
    a gguf, a .gguf that answers with an html error page is not, and
    the first four bytes settle both cases, which reading the spelling
    cannot. Also accepts the shorthand a model card shows: repo owner,
    repo name and file."""
    if looks_remote(arg):
        parts = urllib.parse.urlsplit(arg)
        if parts.netloc.lower() not in HF_HOSTS:
            return arg
        path = parts.path.replace("/blob/", "/resolve/")
        return urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, path, parts.query, ""))
    match = re.match(r"^([\w.-]+)/([\w.-]+)/([\w.@+-]+\.gguf)$", str(arg))
    if match:
        return "https://huggingface.co/{}/{}/resolve/main/{}".format(
            *match.groups())
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
