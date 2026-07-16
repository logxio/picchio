"""Binary evidence capture shared by parity and agent traces."""

import base64
import binascii
import hashlib
import os

from .artifacts import atomic_write_bytes


WIRE_PROVENANCE = {
    "direct_engine_send",
    "product_engine_boundary_capture",
    "application_engine_boundary_capture",
}


def capture_wire_evidence(supplied, attempt_dir, store,
                          relative_path="engine-wire-body.bin"):
    """Persist exact engine-boundary bytes and recompute their digest."""
    if not isinstance(supplied, dict):
        return None
    evidence = {
        "kind": supplied.get("kind"),
        "provenance": supplied.get("provenance"),
        "claimedSha256": supplied.get("sha256"),
        "valid": False,
    }
    encoded = supplied.get("bodyBase64")
    if not isinstance(encoded, str):
        evidence["error"] = "bodyBase64_missing"
        return evidence
    try:
        body = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError):
        evidence["error"] = "bodyBase64_invalid"
        return evidence
    path = os.path.join(attempt_dir, relative_path)
    atomic_write_bytes(path, body)
    digest = hashlib.sha256(body).hexdigest()
    evidence.update({
        "bodyPath": store.relative(path),
        "bodySha256": digest,
        "byteCount": len(body),
        "claimMatches": supplied.get("sha256") == digest,
    })
    if supplied.get("kind") != "actual_engine_request_body":
        evidence["error"] = "kind_not_actual_engine_request_body"
    elif supplied.get("provenance") not in WIRE_PROVENANCE:
        evidence["error"] = "provenance_not_accepted"
    elif not evidence["claimMatches"]:
        evidence["error"] = "sha256_mismatch"
    else:
        evidence["valid"] = True
    return evidence
