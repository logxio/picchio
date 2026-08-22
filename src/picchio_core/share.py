"""picchio share: the postable shapes of a verdict block.

Presentation only. Everything here takes the facts the entry point
already gathered and decides how they read; nothing in this file opens
a file, runs an engine or reaches a network. That split is the point:
the formats are what grow when a new place to post one appears, and
they cannot drift from the block because they never fetch anything of
their own.
"""

# What an absent field prints as. Absent is a reading, not a blank: a
# shorter line with the field quietly dropped reads as though nobody
# had ever asked for it, and the whole reason these formats exist is
# that people kept having to ask.
SHARE_NA = "n/a"
SHARE_URL = "https://github.com/logxio/picchio"

def share_missing(f):
    """The fields that came back with nothing, named so the author can
    close the gap before posting rather than after being asked."""
    return [k for k in ("quant", "bpw", "sha256", "engine", "machine",
                        "ctx", "kv", "settings") if not f.get(k)]


def _share_rate(v):
    return "{:.1f}".format(v) if v is not None else SHARE_NA


def share_line(f):
    """One line with everything on it, for a comment. Long, but it is
    one paragraph in a comment box, and every field on it is one the
    thread this comes from kept having to ask for. A field nobody
    recorded says so; dropping it would make a shorter line that reads
    as though the question was never asked."""
    bits = [f["model"] or SHARE_NA,
            f["quant"] or "quant " + SHARE_NA,
            "{:.2f} bpw".format(f["bpw"]) if f["bpw"]
            else "bpw " + SHARE_NA,
            "sha256 " + (f["sha256"][:12] if f["sha256"] else SHARE_NA),
            f["engine"] or "engine " + SHARE_NA,
            f["machine"] or "machine " + SHARE_NA,
            "ctx {}".format(f["ctx"] or SHARE_NA),
            "KV {}".format(f["kv"] or SHARE_NA),
            f["settings"] or "settings " + SHARE_NA,
            "pp {}".format(_share_rate(f["pp"])),
            "tg {} tok/s".format(_share_rate(f["tg"])),
            "picchio {} {}".format(f["protocol"] or SHARE_NA,
                                    SHARE_URL)]
    return " | ".join(bits)


SHARE_COLUMNS = (("model", "model"), ("quant", "quant"), ("bpw", "bpw"),
                 ("engine", "engine"), ("machine", "machine"),
                 ("ctx", "ctx"), ("kv", "KV"), ("pp", "pp tok/s"),
                 ("tg", "tg tok/s"),
                 ("verdict", "[picchio verdict]({})".format(SHARE_URL)))


def _share_cell(f, key):
    v = f.get(key)
    if v is None:
        return SHARE_NA
    if key == "bpw":
        return "{:.2f}".format(v)
    if key in ("pp", "tg"):
        return _share_rate(v)
    return str(v)


def share_row(f):
    """Header, alignment rule and one row, so the output renders as a
    table on its own wherever it lands rather than only inside someone
    else's. Numeric columns right align, the shape the row library
    already publishes."""
    align = ["---"] * 7 + ["---:", "---:", "---"]
    return "\n".join((
        "| " + " | ".join(lb for _k, lb in SHARE_COLUMNS) + " |",
        "| " + " | ".join(align) + " |",
        "| " + " | ".join(_share_cell(f, k)
                          for k, _lb in SHARE_COLUMNS) + " |"))


def share_post(f, block):
    """A post with the required fields already filled and the opinion
    left blank. The blank is deliberate: picchio measured the run, the
    person had the experience, and the sentence that says whether this
    is good is theirs to write."""
    out = ["## {} on {}".format(f["model"] or SHARE_NA,
                                f["machine"] or SHARE_NA),
           "",
           "<!-- your take goes here: is this good, bad, or expected, "
           "and what were you doing with it -->",
           "",
           "### Specs", ""]
    out.append(share_row(f))
    out += ["", "### The run", "", "```text"]
    out += block.rstrip().splitlines()
    out += ["```", "",
            "Measured with picchio {}, the same fixed prompt three times, "
            "first pass cold: {}".format(
                f["protocol"] or "", SHARE_URL)]
    miss = share_missing(f)
    if miss:
        out += ["", "Not recorded in this run: {}.".format(", ".join(miss))]
    return "\n".join(out)


def render_share(facts, block, mode):
    """Choose one of the three public shapes without duplicating that
    dispatch in the legacy entry point."""
    if mode == "post":
        return share_post(facts, block)
    if mode == "row":
        return share_row(facts)
    return share_line(facts)
