"""picchio vet: what someone else's post states about a tok/s number.

Self-contained on purpose. Nothing here reads a model, a log or a
machine; it reads text and reports what is in it. The verdict block
wrap width is passed in rather than redefined, so the block format
keeps one definition."""

import re
import sys
import textwrap


# picchio vet reads someone else's post and answers the question that
# thread asked over and over by hand: which of the numbers that move
# this result did they actually state. It reports what is there, what is
# not, and hands back a template the author can fill in. It never scores
# the model, never scores the person, and never guesses a value that is
# not written down: a field it cannot read is unparsed, which is a
# different thing from absent and both are different from wrong.

# Each field is (key, label, pattern). The pattern's `v` group is the
# value; a match with no `v` means the subject was raised without a
# readable value, which reads back as unparsed rather than as declared.
VET_FIELDS = (
    # either a parameter-size token ("Qwen3.5-9B", "gpt-oss 20B") or a
    # registry slug. Both sides of the slash need three characters, or
    # "50 t/s" reads as a repository called t owned by nobody.
    ("model", "model", r"\b(?P<v>[\w.]+(?:[-_][\w.]+)*[-_ ]\d+(?:\.\d+)?"
                       r"\s?[Bb]|[\w.-]{3,}/[\w.:-]{3,})\b"),
    ("quant", "quant recipe",
     r"\b(?P<v>I?Q\d+(?:_[A-Za-z0-9]+)+|BF16|F16|F32|MXFP4|NVFP4)\b"),
    ("bpw", "effective bits per weight",
     r"(?P<v>\d+(?:\.\d+)?)\s*(?:bpw|bits per weight)"),
    ("identity", "file identity or uploader",
     r"(?:sha256|huggingface\.co/|(?P<v2>[\w.-]+/[\w.-]*GGUF[\w.:-]*))"
     r"\s*(?P<v>[0-9a-f]{8,}|[\w.:/-]+)?"),
    ("engine", "engine",
     r"\b(?P<v>llama\.cpp|llama-server|llama-bench|llama-cli|ollama|"
     r"lm ?studio|vllm|mlx[\w-]*|koboldcpp|kobold|exllama\w*|llamafile|"
     r"text-generation-webui)\b"),
    ("engine_version", "engine version",
     r"\b(?P<v>b\d{4,5}|\d+\.\d+\.\d+)\b"),
    ("machine", "machine or gpu",
     r"\b(?P<v>RTX ?\d{3,4}\w*|GTX ?\d{3,4}|RX ?\d{4}\w*|Radeon[\w ]*|"
     r"M[1-5](?: ?(?:Pro|Max|Ultra))?|EPYC|Ryzen|Xeon|"
     r"\d+ ?GB(?: VRAM| RAM)?)\b"),
    # -c has to stand alone as a flag: -ctk and -ctv start with it and
    # are a different setting entirely, and reading "-ctk q8_0" as a
    # context of 8 is worse than reading nothing
    ("ctx", "context length",
     r"(?P<v2>\d[\d,]*k?)\s*(?:tokens?\s+of\s+)?(?:ctx|context)\b"
     r"|(?:\bctx\b|\bcontext\b|\bn_ctx\b|(?<![\w-])-c(?=\s))"
     r"\D{0,12}(?P<v>\d[\d,]*k?)?"),
    ("kv", "kv cache dtype",
     r"(?:-ctk|-ctv|type_k|type_v|kv[ _-]?cache)\s*"
     r"(?P<v>[fqi]\d+[\w_]*)?"),
    ("sampling", "sampling settings",
     r"(?:temp(?:erature)?|top[_-]?[kp]|min[_-]?p|\bseed)\b\D{0,4}"
     r"(?P<v>[\d.]+)?"),
    ("placement", "gpu placement",
     r"(?:-ngl|n_gpu_layers|\bngl\b|offload\w*|layers on gpu)\s*"
     r"(?P<v>\d+)?"),
    # the bare unit is its own alternative: a llama-bench table puts
    # "t/s" in a column header and the figures in cells three pipes
    # away, which is a rate that was stated and cannot be read off the
    # text. That is unparsed, and unparsed is not the same as absent.
    ("rate", "rate",
     r"(?P<v>\d+(?:\.\d+)?)\s*(?:tok/s|tokens?/s|t/s|tps)\b"
     r"|(?:tok/s|tokens?/s|t/s|tps)\b"),
)

# The known mechanisms, named only when the text raises them. Each one
# cites a file in this repository that measured it or that shows the
# mechanism being read, because the alternative is a tool lecturing
# from memory. A mechanism nobody here has measured gets no entry,
# however well known it is; `sources` is checked against the tree by
# the selftest. Every note says what was measured. None of them says
# whether a model, a setting or a person is any good.
VET_NOTES = (
    {"key": "low-bit",
     "trigger": r"\b(I?Q[123]\b(?!_)|IQ[123]_\w+|Q[123]_\w+|"
                r"[123][ -]bit)\b",
     "note": "a low bit class label is a mix, not a width: q2_K prices "
             "at 2.63 bits per weight and iq1_s at 1.56 before the "
             "higher precision tensors every recipe keeps, so the "
             "label alone does not say what was kept",
     "sources": ("picchio.py",)},
    {"key": "kv-dtype",
     "trigger": r"(?:-ctk|-ctv|type_[kv]|kv[ _-]?cache)\D{0,12}"
                r"[qi]\d|kv[ _-]?cache\s*(?:quant|at)",
     "note": "a quantized kv cache is a runtime flag that never "
             "reaches the model file, so it has to be stated "
             "separately from the weight quant; picchio reads the "
             "engine's own kv line for it",
     "sources": ("examples/raw/kv-q8/ctk-q8.stderr.txt",)},
    {"key": "ctx-depth",
     "trigger": r"\b(?:1[6-9]|[2-9]\d|\d{3,})k\b|"
                r"\b(?:1[6-9]|[2-9]\d)\d{3,}\b|truncat|shift",
     "note": "context depth is paid for in decode: the same build fell "
             "11% in decode, 20.0 to 17.9 tok/s, between 2,531 and "
             "21,079 tokens in context, and end to end fell 5x",
     "sources": ("examples/ctx-sweep.txt",)},
    {"key": "quant-spread",
     "trigger": r"\bI?Q\d",
     "note": "two files under one quant label are not one file: four "
             "published Q4_K_M builds of the same model priced out "
             "between 5.02 and 5.27 effective bits per weight here",
     "sources": ("examples/quantizers/README.md",)},
    {"key": "placement",
     "trigger": r"-ngl|n_gpu_layers|\boffload|\bngl\b|layers on gpu",
     "note": "placement moves prefill far more than decode: the same "
             "model, machine and engine measured 588.0 and 21.1 tok/s "
             "on the gpu against 26.8 and 12.2 on the cpu, so a decode "
             "figure alone barely shows it",
     "sources": ("examples/healthy-metal.txt",
                 "examples/cpu-fallback.txt")},
    {"key": "moe-residency",
     "trigger": r"\bA\d+B\b|mixture of experts|\bMoE\b|\bexperts?\b",
     "note": "a mixture of experts can hold far less than its weights "
             "file and pay for it in speed: 20.6 GiB of weights held "
             "at 9.8 GiB decoded 11 tok/s against 30, and took 24.3 s "
             "to first token against 1.3",
     "sources": ("README.md",)},
)


def vet_scan(text):
    """{field key: value or "unparsed"} for everything the text states.

    Every match is walked, not just the leftmost, and the first one
    carrying a value wins: a post that says "kv cache" in a sentence
    before it says "-ctk q8_0" has stated q8_0, and stopping at the
    first mention would report it as unreadable. A subject raised with
    no readable value anywhere is unparsed, which is its own answer and
    not the same as never mentioned. Nothing is inferred across fields:
    naming a gpu is not claiming a context length."""
    found = {}
    for key, _label, pat in VET_FIELDS:
        seen = False
        for m in re.finditer(pat, text, re.I):
            seen = True
            v = m.groupdict().get("v") or m.groupdict().get("v2")
            if v and v.strip():
                found[key] = v.strip()
                break
        if seen and key not in found:
            found[key] = "unparsed"
    return found


def vet_quant_note(text, found):
    """A bare quant class is not a quant recipe. "Q4" names a family
    with several members that price differently, and the gap between
    them is the thing the thread this comes from was arguing about."""
    if "quant" in found:
        return None
    m = re.search(r"\b(I?Q\d)\b(?!_)", text, re.I)
    return m.group(1).upper() if m else None


def vet_rate_lane(text):
    """Which of the three rates a number is, when the post says. A bare
    tok/s figure is the ambiguity picchio exists to name, so this
    reports the ambiguity rather than picking a lane."""
    if re.search(r"\b(pp\d*|prefill|prompt eval|prompt processing)\b",
                 text, re.I) and re.search(
                     r"\b(tg\d*|decode|eval|generation|token gen)\b",
                     text, re.I):
        return "both prefill and decode"
    if re.search(r"\b(pp\d*|prefill|prompt eval)\b", text, re.I):
        return "prefill only"
    if re.search(r"\b(tg\d*|decode|generation)\b", text, re.I):
        return "decode only"
    return None


def _wrap(text, width):
    """Card prose at the block's own width, so vet output pastes into
    the same places a verdict block does."""
    return textwrap.wrap(text, width=width - 2,
                         initial_indent="  ", subsequent_indent="  ")


def render_vet(found, quant_class, lane, notes, missing, width):
    out = ["picchio vet: {} of {} fields stated".format(
        len(found), len(VET_FIELDS))]
    labels = {k: lb for k, lb, _p in VET_FIELDS}
    if found:
        out += ["", "  stated"]
        for key, label, _pat in VET_FIELDS:
            if key in found:
                out.append("    {:<26}{}".format(label, found[key]))
    if quant_class:
        out += ["", "  {} names a family, not a file. Its members price "
                    "differently;".format(quant_class),
                "  the recipe (Q4_K_M, Q4_K_S, UD-Q4_K_XL) is the part "
                "that identifies one."]
    if "rate" in found:
        out += [""]
        out += _wrap(
            "the rate is labeled {}.".format(lane) if lane else
            "the rate carries no lane. Prefill, decode and end to end "
            "differ by an order of magnitude on one machine, so an "
            "unlabeled figure names a speed without saying of what.",
            width)
    if missing:
        out += ["", "  not stated"]
        for key in missing:
            out.append("    " + labels[key])
    if notes:
        out += ["", "  mechanisms this post raises, measured here"]
        for n in notes:
            out += _wrap(n["note"], width)
            out.append("      see {}".format(", ".join(n["sources"])))
    out += ["", "  a reply they can fill in", "", "  ```"]
    for _key, label, _pat in VET_FIELDS:
        out.append("  {}:".format(label))
    out.append("  ```")
    return "\n".join(out)


def vet_cli(argv, width):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio vet [FILE]\n"
              "read a post or comment and list which of the settings\n"
              "that move a tok/s number it actually states, which it\n"
              "does not, and the mechanisms behind the ones it skipped,\n"
              "each with the measurement in this repo that showed it.\n"
              "Ends with a template the author can fill in. FILE or\n"
              "stdin. It does not judge the model, score the post, or\n"
              "fill in a value that is not written down.")
        sys.exit(0)
    if len(argv) > 1:
        sys.exit("picchio vet: usage: picchio vet [FILE]")
    try:
        text = sys.stdin.read() if not argv or argv[0] == "-" \
            else open(argv[0]).read()
    except OSError as e:
        sys.exit("picchio vet: {}".format(e))
    if not text.strip():
        sys.exit("picchio vet: nothing to read on stdin. Paste a post "
                 "or pass a file.")
    found = vet_scan(text)
    quant_class = vet_quant_note(text, found)
    missing = [k for k, _lb, _p in VET_FIELDS if k not in found]
    notes = [n for n in VET_NOTES
             if re.search(n["trigger"], text, re.I)]
    print(render_vet(found, quant_class, vet_rate_lane(text), notes,
                     missing, width))
    sys.exit(0)
