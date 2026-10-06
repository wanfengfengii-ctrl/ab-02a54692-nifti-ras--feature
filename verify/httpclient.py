"""Minimal multipart/form-data client helpers (stdlib only)."""
from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

BOUNDARY = "nifti-verify-7f3a9c51e2b44d08a1"


def build_multipart(file_bytes, payload, *, file_field="file",
                    filename="vol.nii", payload_field="points"):
    """Assemble a multipart/form-data body with one file part and one
    JSON field (``points`` or ``regions``).  ``payload`` may be a Python
    object (JSON-encoded) or raw bytes/str (sent as-is, for malformed
    input tests)."""
    if not isinstance(payload, (bytes, str)):
        payload = json.dumps(payload)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    boundary = BOUNDARY.encode("ascii")
    return b"\r\n".join([
        b"--" + boundary,
        b'Content-Disposition: form-data; name="%s"; filename="%s"'
        % (file_field.encode("utf-8"), filename.encode("utf-8")),
        b"Content-Type: application/octet-stream",
        b"",
        file_bytes,
        b"--" + boundary,
        b'Content-Disposition: form-data; name="%s"' % payload_field.encode("utf-8"),
        b"Content-Type: application/json",
        b"",
        payload,
        b"--" + boundary + b"--",
        b"",
    ])


def _post(base_url, path, file_bytes, payload, *, payload_field="points",
          timeout=30):
    url = urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        conn.request(
            "POST", path,
            body=build_multipart(file_bytes, payload,
                                 payload_field=payload_field),
            headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    finally:
        conn.close()
    try:
        return status, json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return status, None


def post_sample(base_url, file_bytes, points, *, timeout=30):
    """POST /api/nifti/sample; returns ``(status, parsed_json_or_None)``."""
    return _post(base_url, "/api/nifti/sample", file_bytes, points,
                 payload_field="points", timeout=timeout)


def post_sphere_stats(base_url, file_bytes, regions, *, timeout=30):
    """POST /api/nifti/sphere-stats; returns
    ``(status, parsed_json_or_None)``."""
    return _post(base_url, "/api/nifti/sphere-stats", file_bytes, regions,
                 payload_field="regions", timeout=timeout)


def get_json(base_url, path, *, timeout=5):
    """GET a JSON document; returns ``(status, parsed_json_or_None)``."""
    url = urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    finally:
        conn.close()
    try:
        return status, json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return status, None
