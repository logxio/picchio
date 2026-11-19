"""Mechanical grader-touch candidates with surviving PR-level line edits.

All Smoke (figshare 10.6084/m9.figshare.32032107.v1) is the source of the
W1:S3 assertion patterns below. Its released classifier scores additions to
an entire file patch; here the same expressions score individual surviving
lines so a deletion can be compared with a replacement in one test function.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

NAME = re.compile(r"(^|[\\/])(tests?|testing|cypress)([\\/]|$)", re.I)
TOKEN = re.compile(r"(^|[\\/_.-])(test|spec)([\\/_.-]|$)", re.I)
SUFFIX = re.compile(r"(Test|Spec)($|\.)")
EXCLUDE = {".csv", ".doc", ".json", ".md", ".mk", ".rtf", ".txt", ".yaml", ".yml"}
CODE = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".go", ".java", ".kt", ".rb", ".rs", ".cs", ".c", ".cpp", ".cc", ".h", ".hpp", ".swift", ".php", ".dart", ".lua", ".ex", ".exs", ".zig", ".scala", ".clj", ".r", ".m", ".mm", ".fs", ".fsx", ".sh", ".bash", ".vue", ".svelte"}
DATA_PART = re.compile(r"(^|/)(fixture|fixtures|snapshot|snapshots|__snapshots__|golden|testdata|test_data)(/|$)", re.I)
DATA_SUFFIX = re.compile(r"(?:\.snap$|\.golden$|\.expected$|\.error$|\.out$|\.fixture$)", re.I)
CI = re.compile(r"(^|/)(\.github/workflows/[^/]+|tox\.ini|pytest\.ini|setup\.cfg|\.coveragerc|jest\.config(?:\.[^/]+)?|codecov\.ya?ml)$", re.I)
SKIP = re.compile(r"@pytest\.mark\.(?:skip|xfail)\b|@unittest\.skip\b|\b(?:it|describe|test)\.skip\s*\(|\b(?:xit|xdescribe)\s*\(|\bt\.Skip(?:f|Now)?\s*\(|#\[ignore\]|@Disabled\b|@Ignore\b|pytest\.skip\s*\(", re.I)
ASSERT = re.compile(r"\bassert\b|\bassert[A-Z_a-z]+\b|\bexpect\s*\(|\brequire\.[A-Za-z]+|\bt\.(?:Errorf?|Fatalf?)\s*\(|\bverify\s*\(", re.I)
FN = re.compile(r"\b(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\b|\bfunc\s+(Test[A-Za-z0-9_]+)\s*\(|\b(?:function\s+)?(test[A-Z][A-Za-z0-9_]+)\s*\(|\b(?:it|test|describe)\s*\(\s*['\"`]([^'\"`]+)", re.I)
LITERAL = re.compile(r"(?<![\w.])(?:-?\d+(?:\.\d+)?(?:e[-+]?\d+)?|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')(?!\w)", re.I)
TOL = re.compile(r"\b(atol|rtol|delta|tolerance|tol|abs|rel|places)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
THRESHOLD = re.compile(r"(?:\bassert\b|\bexpect\s*\().*?(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
FAIL_UNDER = re.compile(r"(?:fail_under|fail-under|cov-fail-under)\s*[=: ]\s*(\d+(?:\.\d+)?)", re.I)
TEST_JOB = re.compile(r"^\s*([\w.-]*(?:test|check|lint|coverage)[\w.-]*)\s*:\s*(?:#.*)?$", re.I)

# Expressions released in All Smoke's 00_data_preparation.py, normalized to a
# single changed line. W1 means no recognized assertion type.
ALL_SMOKE = {
    "existence": re.compile(r"assertNotNull|assertIsNotNone|\.toBeDefined|\.not\.toBeNull|assertNotNil", re.I),
    "boolean": re.compile(r"assertTrue|assertFalse|assert\s+(True|False)|\.toBeTruthy|\.toBeFalsy", re.I),
    "equality": re.compile(r"assertEqual|assertEquals|assert_eq|expect.*\.toBe\(|expect.*\.toEqual\(|expect.*to\.equal|assert\s+.*==|Assert\.Equal|assert_equal", re.I),
    "comparison": re.compile(r"assertGreater|assertLess|\.toBeGreater|\.toBeLess|\.above|\.below", re.I),
    "contain": re.compile(r"assertIn|assertContains|assert.*\bin\b|\.toContain|\.include|assert_includes", re.I),
    "error": re.compile(r"assertRaises|assertThrows|assert_raises|\.toThrow|\.rejects|pytest\.raises|should_panic", re.I),
    "typecheck": re.compile(r"assertIsInstance|isinstance|\.toBeInstanceOf|assert_kind_of", re.I),
    "mock": re.compile(r"\.toHaveBeenCalled|\.assert_called|\.called_with|verify\(.*\)\.called|expect.*mock", re.I),
    "snapshot": re.compile(r"\.toMatchSnapshot|\.toMatchInlineSnapshot", re.I),
    "go": re.compile(r"t\.Error[f]?\(|t\.Fatal[f]?\(|require\.\w+|assert\.\w+\("),
}
STRENGTH = {"W1": 1, "W2": 2, "W3": 3, "W4": 4, "W5": 5, "S1": 6, "S2": 7, "S3": 8}


def is_ci_file(path: str) -> bool:
    return bool(CI.search(path or ""))


def is_test_code(path: str) -> bool:
    path = (path or "").replace("\\", "/")
    ext = PurePosixPath(path.lower()).suffix
    return (ext not in EXCLUDE and not DATA_PART.search(path)
            and not DATA_SUFFIX.search(path) and bool(NAME.search(path) or TOKEN.search(path) or SUFFIX.search(path)))


def is_source(path: str) -> bool:
    path = (path or "").replace("\\", "/")
    return (PurePosixPath(path.lower()).suffix in CODE and not is_test_code(path)
            and not is_ci_file(path) and not DATA_PART.search(path))


def oracle_class(line: str) -> str:
    found = {name for name, pattern in ALL_SMOKE.items() if pattern.search(line)}
    strong = found & {"equality", "comparison", "contain", "error", "typecheck", "go"}
    if len(strong) >= 2: return "S3"
    if found & {"error", "typecheck", "contain"}: return "S2"
    if found & {"equality", "comparison", "go"}: return "S1"
    if "snapshot" in found: return "W5"
    if "mock" in found: return "W4"
    if "boolean" in found: return "W3"
    if "existence" in found: return "W2"
    return "W1"


def function_name(line: str) -> str | None:
    m = FN.search(line)
    if not m: return None
    return next((x for x in m.groups() if x), None)


@dataclass
class Line:
    text: str
    sha: str
    hunk: str
    fn: str | None

    def proof(self, path: str) -> dict:
        return {"path": path, "sha": self.sha, "line": self.text, "function": self.fn,
                "hunk": "\n".join(self.hunk.splitlines()[:120])}


@dataclass
class NetFile:
    path: str
    statuses: list[str] = field(default_factory=list)
    adds: dict[str, list[Line]] = field(default_factory=lambda: defaultdict(list))
    dels: dict[str, list[Line]] = field(default_factory=lambda: defaultdict(list))
    sample_hunk: str = ""

    def change(self, kind: str, line: Line) -> None:
        mine, opposite = (self.adds, self.dels) if kind == "+" else (self.dels, self.adds)
        stack = opposite[line.text]
        for i in range(len(stack) - 1, -1, -1):
            other = stack[i]
            if not (line.fn and other.fn and line.fn != other.fn):
                stack.pop(i)
                break
        else:
            mine[line.text].append(line)

    def lines(self, kind: str) -> list[Line]:
        target = self.adds if kind == "+" else self.dels
        return [x for stack in target.values() for x in stack]


def add_patch(net: NetFile, sha: str, status: str, patch: str | None) -> None:
    net.statuses.append((status or "").lower())
    if not patch: return
    chunks = re.split(r"(?=^@@[^\n]*@@)", patch, flags=re.M)
    for chunk in chunks:
        if not chunk.startswith("@@"): continue
        hunk = "\n".join(chunk.splitlines()[:120])
        if not net.sample_hunk: net.sample_hunk = hunk
        fn = function_name(chunk.splitlines()[0])
        for raw in chunk.splitlines()[1:]:
            if not raw or raw[0] not in "+- ": continue
            if raw.startswith("+++ ") or raw.startswith("--- "): continue
            if raw[0] == " ":
                fn = function_name(raw[1:]) or fn
            else:
                text = raw[1:]
                line_fn = function_name(text) or fn
                net.change(raw[0], Line(text, sha, hunk, line_fn))
                fn = line_fn


def _same_fn(a: Line, b: Line) -> bool:
    if a.fn and b.fn: return a.fn == b.fn
    return a.hunk == b.hunk


def _literal_pair(a: str, b: str) -> bool:
    if a == b or not ASSERT.search(a) or not ASSERT.search(b): return False
    aa, bb = LITERAL.findall(a), LITERAL.findall(b)
    return bool(aa and bb and aa != bb and LITERAL.sub("<lit>", a) == LITERAL.sub("<lit>", b))


def _source_sync(a: str, b: str, source_pairs: set[tuple[str, str]]) -> bool:
    old, new = LITERAL.findall(a), LITERAL.findall(b)
    return any((x, y) in source_pairs for x, y in zip(old, new) if x != y)


def _looser(a: str, b: str) -> bool:
    x, y = TOL.search(a), TOL.search(b)
    if x and y and x.group(1).lower() == y.group(1).lower():
        old, new = float(x.group(2)), float(y.group(2))
        return new < old if x.group(1).lower() == "places" else new > old
    x, y = THRESHOLD.search(a), THRESHOLD.search(b)
    if x and y and x.group(1)[0] == y.group(1)[0]:
        old, new = float(x.group(2)), float(y.group(2))
        if x.group(1)[0] == "<":
            return new > old or (new == old and x.group(1) == "<" and y.group(1) == "<=")
        return new < old or (new == old and x.group(1) == ">" and y.group(1) == ">=")
    return False


def grade_pr(rows: list[dict], test_path=None) -> dict:
    """Rows are already ordered by commit sequence; one PR, arbitrary files."""
    test_path = test_path or is_test_code
    files: dict[str, NetFile] = {}
    missing = False
    raw_size = 0
    for row in rows:
        path = row["filename"] or ""
        if not path: continue
        net = files.setdefault(path, NetFile(path))
        patch = row.get("patch")
        missing |= patch is None
        raw_size += int((row.get("additions") or 0) + (row.get("deletions") or 0))
        add_patch(net, row.get("sha") or "", row.get("status") or "", patch)
    src = [f for f in files.values() if is_source(f.path) and not test_path(f.path)
           and (f.lines("+") or f.lines("-"))]
    tests = [f for f in files.values() if test_path(f.path)]
    cis = [f for f in files.values() if is_ci_file(f.path)]
    source_pairs: set[tuple[str, str]] = set()
    for f in src:
        added_by_shape: dict[str, list[Line]] = defaultdict(list)
        for b in f.lines("+"):
            if LITERAL.search(b.text): added_by_shape[LITERAL.sub("<lit>", b.text)].append(b)
        for a in f.lines("-"):
            for b in added_by_shape.get(LITERAL.sub("<lit>", a.text), []):
                source_pairs.update((x,y) for x,y in zip(LITERAL.findall(a.text),LITERAL.findall(b.text)) if x != y)
    flags = {f"T{i}": False for i in range(1, 7)}
    evidence: dict[str, list[dict]] = {k: [] for k in flags}
    added_names = {function_name(x.text) for f in tests for x in f.lines("+")}
    added_basenames = {PurePosixPath(f.path).name for f in tests if f.lines("+")}
    added_file_signatures = {tuple(sorted(Counter(x.text for x in f.lines("+")).items()))
                             for f in tests if f.statuses and f.statuses[-1] in ("added", "renamed")}
    for f in tests:
        removed, added = f.lines("-"), f.lines("+")
        moved = (f.statuses and f.statuses[-1] == "removed" and removed
                 and tuple(sorted(Counter(x.text for x in removed).items())) in added_file_signatures)
        if moved: continue
        assert_by_fn: dict[str, int] = defaultdict(int)
        assert_by_hunk: dict[str, int] = defaultdict(int)
        assert_unknown_fn_by_hunk: dict[str, int] = defaultdict(int)
        literals_by_shape: dict[str, list[Line]] = defaultdict(list)
        tolerance_by_name: dict[str, list[Line]] = defaultdict(list)
        threshold_by_op: dict[str, list[Line]] = defaultdict(list)
        for b in added:
            if ASSERT.search(b.text):
                rank = STRENGTH[oracle_class(b.text)]
                if b.fn: assert_by_fn[b.fn] = max(assert_by_fn[b.fn], rank)
                else: assert_unknown_fn_by_hunk[b.hunk] = max(assert_unknown_fn_by_hunk[b.hunk], rank)
                assert_by_hunk[b.hunk] = max(assert_by_hunk[b.hunk], rank)
                if LITERAL.search(b.text):
                    literals_by_shape[LITERAL.sub("<lit>", b.text)].append(b)
            if m := TOL.search(b.text): tolerance_by_name[m.group(1).lower()].append(b)
            if m := THRESHOLD.search(b.text): threshold_by_op[m.group(1)[0]].append(b)
        for a in removed:
            if ASSERT.search(a.text):
                rank = STRENGTH[oracle_class(a.text)]
                replacement_rank = (max(assert_by_fn[a.fn], assert_unknown_fn_by_hunk[a.hunk])
                                    if a.fn else assert_by_hunk[a.hunk])
                if replacement_rank < rank:
                    flags["T1"] = True; evidence["T1"].append(a.proof(f.path))
            if function_name(a.text) and function_name(a.text) not in added_names:
                flags["T5"] = True; evidence["T5"].append(a.proof(f.path))
            candidates = []
            if m := TOL.search(a.text): candidates.extend(tolerance_by_name[m.group(1).lower()])
            if m := THRESHOLD.search(a.text):
                candidates.extend(threshold_by_op[m.group(1)[0]])
            if ASSERT.search(a.text) and LITERAL.search(a.text):
                candidates.extend(literals_by_shape[LITERAL.sub("<lit>", a.text)])
            for b in candidates:
                if not _same_fn(a,b): continue
                if _looser(a.text,b.text):
                    flags["T4"] = True; evidence["T4"].append({"before":a.proof(f.path),"after":b.proof(f.path)})
                if (_literal_pair(a.text,b.text) and not _source_sync(a.text,b.text,source_pairs)
                        and not (TOL.search(a.text) or TOL.search(b.text)
                                 or THRESHOLD.search(a.text) or THRESHOLD.search(b.text))):
                    flags["T3"] = True; evidence["T3"].append({"before":a.proof(f.path),"after":b.proof(f.path)})
        for b in added:
            if SKIP.search(b.text):
                flags["T2"] = True; evidence["T2"].append(b.proof(f.path))
        if f.statuses and f.statuses[-1] == "removed" and PurePosixPath(f.path).name not in added_basenames:
            removed_names = {function_name(a.text) for a in removed if function_name(a.text)}
            if (removed and not any(s == "renamed" for s in f.statuses)
                    and not (removed_names & added_names)):
                flags["T5"] = True; evidence["T5"].append(removed[0].proof(f.path))
    for f in cis:
        removed, added = f.lines("-"), f.lines("+")
        for b in added:
            value = b.text
            if re.search(r"\bcontinue-on-error\s*:\s*true\b|\ballow_failures\s*[:=](?!\s*false\b)",value,re.I):
                flags["T6"] = True; evidence["T6"].append(b.proof(f.path))
            y = FAIL_UNDER.search(value)
            if y:
                for a in removed:
                    x = FAIL_UNDER.search(a.text)
                    if x and float(y.group(1)) < float(x.group(1)):
                        flags["T6"] = True; evidence["T6"].append({"before":a.proof(f.path),"after":b.proof(f.path)})
        added_jobs = {m.group(1).lower() for b in added if (m:=TEST_JOB.search(b.text))}
        for a in removed:
            m = TEST_JOB.search(a.text)
            if m and m.group(1).lower() not in added_jobs:
                flags["T6"] = True; evidence["T6"].append(a.proof(f.path))
    return {"flags": flags, "evidence": {k:v[:8] for k,v in evidence.items() if v},
            "has_source": bool(src), "source_paths": [f.path for f in src],
            "has_test": bool(tests), "test_paths": [f.path for f in tests],
            "test_hunk": next((f.sample_hunk for f in tests if f.sample_hunk), ""),
            "missing_patch": missing, "size": raw_size,
            "grader_touch": bool(src) and any(flags.values())}
