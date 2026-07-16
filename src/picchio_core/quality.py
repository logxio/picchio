"""Quality is an independent verdict, never inferred from runtime success."""

import json
import os
import re

from .adapters import AdapterError, command_json


def _path_exists(value, dotted):
    current = value
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() \
                and int(part) < len(current):
            current = current[int(part)]
        else:
            return False
    return True


def evaluate_builtin(rubric, text, raw):
    if not rubric:
        return None
    failures = []
    contains = list(rubric.get("contains", []))
    if rubric.get("codeword"):
        contains.append(rubric["codeword"])
    for needle in contains:
        if str(needle) not in text:
            failures.append("missing text: {}".format(needle))
    for needle in rubric.get("notContains", []):
        if str(needle) in text:
            failures.append("forbidden text: {}".format(needle))
    for pattern in rubric.get("regex", []):
        try:
            matched = re.search(pattern, text) is not None
        except re.error as exc:
            failures.append("invalid regex {}: {}".format(pattern, exc))
        else:
            if not matched:
                failures.append("regex did not match: {}".format(pattern))
    parsed = None
    if rubric.get("jsonFields"):
        try:
            parsed = json.loads(text)
        except ValueError:
            failures.append("model text is not JSON")
    for path in rubric.get("jsonFields", []):
        if parsed is not None and not _path_exists(parsed, path):
            failures.append("missing JSON field: {}".format(path))
    for path in rubric.get("responseFields", []):
        if not _path_exists(raw, path):
            failures.append("missing response field: {}".format(path))
    return {"status": "fail" if failures else "pass",
            "source": "rubric", "reasons": failures}


def evaluate_quality(case, response, attempt_dir):
    checks = []
    builtin = evaluate_builtin(case.get("rubric"), response.get("text", ""),
                               response.get("raw", {}))
    if builtin:
        checks.append(builtin)
    reported = response.get("reportedQuality")
    if reported:
        status = str(reported.get("status", "")).lower()
        checks.append({"status": status if status in ("pass", "fail")
                       else "fail", "source": "adapter",
                       "reasons": reported.get("reasons", [])})
    judge = case.get("judge")
    if judge:
        payload = {
            "schema": "picchio.judge.request.v1",
            "caseId": case.get("id"),
            "request": case.get("request"),
            "text": response.get("text", ""),
            "rawResponse": response.get("raw", {}),
        }
        try:
            value, transport = command_json(judge, payload, attempt_dir,
                                            prefix="judge")
            status = str(value.get("status", "")).lower()
            checks.append({
                "status": status if status in ("pass", "fail") else "fail",
                "source": "external_judge",
                "score": value.get("score"),
                "reasons": value.get("reasons", []),
                "rawResponsePath": transport.get("rawResponsePath"),
            })
        except AdapterError as exc:
            checks.append({"status": "fail", "source": "external_judge",
                           "reasons": [str(exc)]})
    if not checks:
        return {"status": "not_evaluated", "checks": []}
    status = "fail" if any(row["status"] != "pass" for row in checks) \
        else "pass"
    return {"status": status, "checks": checks}
