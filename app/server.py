"""HTTP server exposing ``POST /api/nifti/sample`` and
``POST /api/nifti/sphere-stats`` (pure stdlib).

The multipart form carries exactly one NIfTI-1 .nii file (<= 16 MiB) and a
JSON field: ``points`` with 1..256 uniquely-numbered finite 3D coordinates
for sampling, or ``regions`` with 1..32 uniquely-numbered closed balls
(RAS centre + positive finite radius) for local intensity statistics.
See README.md for the full API contract.
"""
from __future__ import annotations

import json
import logging
import math
import os
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .errors import ApiError, PointError, RegionError
from .nifti import NiftiVolume, parse_nifti
from .sampling import sample_point
from .spheres import sphere_stats

log = logging.getLogger("nifti_sampler")

MAX_FILE_BYTES = 16 * 1024 * 1024                # 16 MiB single-file limit
MAX_BODY_BYTES = MAX_FILE_BYTES + 1024 * 1024    # file + multipart overhead
MAX_POINTS = 256
MAX_REGIONS = 32
SAMPLE_PATH = "/api/nifti/sample"
SPHERE_PATH = "/api/nifti/sphere-stats"
HEALTH_PATH = "/healthz"


# ---------------------------------------------------------------------------
# multipart parsing
# ---------------------------------------------------------------------------

class _Part:
    __slots__ = ("name", "filename", "content")

    def __init__(self, name, filename, content):
        self.name = name
        self.filename = filename
        self.content = content


def _parse_disposition(raw):
    if not raw.lower().startswith(b"form-data"):
        raise ApiError("invalid_multipart", "part Content-Disposition is not form-data")
    params = {}
    for token in raw.split(b";")[1:]:
        if b"=" not in token:
            continue
        key, value = token.split(b"=", 1)
        value = value.strip()
        if len(value) >= 2 and value[:1] == b'"' and value[-1:] == b'"':
            value = value[1:-1]
        params[key.strip().lower().decode("ascii", "replace")] = \
            value.decode("utf-8", "replace")
    return params


def parse_multipart(body, boundary):
    """Split a multipart/form-data body into ``_Part`` objects (strict)."""
    try:
        delim = b"--" + boundary.encode("ascii")
    except UnicodeEncodeError:
        raise ApiError("invalid_multipart", "multipart boundary is not ASCII")
    segments = body.split(delim)
    if len(segments) < 3 or segments[0] != b"":
        raise ApiError("invalid_multipart", "malformed multipart body")
    if segments[-1] not in (b"--", b"--\r\n"):
        raise ApiError("invalid_multipart", "missing multipart closing boundary")
    parts = []
    for seg in segments[1:-1]:
        if not (seg.startswith(b"\r\n") and seg.endswith(b"\r\n")):
            raise ApiError("invalid_multipart", "malformed multipart part framing")
        seg = seg[2:-2]
        head, sep, content = seg.partition(b"\r\n\r\n")
        if not sep:
            raise ApiError("invalid_multipart", "part is missing a header/body separator")
        headers = {}
        for line in head.split(b"\r\n"):
            name, sep, value = line.partition(b":")
            if not sep:
                raise ApiError("invalid_multipart", "malformed part header")
            headers[name.strip().lower()] = value.strip()
        params = _parse_disposition(headers.get(b"content-disposition", b""))
        if "name" not in params:
            raise ApiError("invalid_multipart", "part is missing a form field name")
        parts.append(_Part(params["name"], params.get("filename"), content))
    return parts


# ---------------------------------------------------------------------------
# points field parsing
# ---------------------------------------------------------------------------

def parse_points(raw):
    """Validate the ``points`` JSON field; returns ``[(id, (x, y, z)), ...]``
    in request order."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ApiError("invalid_points", "points field is not valid JSON", field="points")
    if not isinstance(data, list):
        raise ApiError("invalid_points", "points must be a JSON array of objects",
                       field="points")
    if not 1 <= len(data) <= MAX_POINTS:
        raise ApiError("invalid_points",
                       f"expected between 1 and {MAX_POINTS} points, got {len(data)}",
                       field="points")
    seen = set()
    points = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ApiError("invalid_points", f"point #{index} is not a JSON object",
                           field="points")
        pid = item.get("id")
        if isinstance(pid, bool) or not isinstance(pid, int) or not 0 <= pid < 2 ** 31:
            raise ApiError("invalid_points",
                           f"point #{index} has invalid id {pid!r}; need an integer "
                           f"in [0, 2^31)",
                           field="points")
        if pid in seen:
            raise ApiError("invalid_points", f"duplicate point id {pid}", field="points")
        seen.add(pid)
        coord = item.get("point")
        if not isinstance(coord, list) or len(coord) != 3:
            raise ApiError("invalid_points",
                           f"point id {pid} needs \"point\": [x, y, z]", field="points")
        xyz = []
        for c in coord:
            if isinstance(c, bool) or not isinstance(c, (int, float)) \
                    or not math.isfinite(c):
                raise ApiError("invalid_points",
                               f"point id {pid} has a non-finite or non-numeric "
                               f"coordinate {c!r}",
                               field="points")
            xyz.append(float(c))
        points.append((pid, tuple(xyz)))
    return points


# ---------------------------------------------------------------------------
# regions field parsing
# ---------------------------------------------------------------------------

def parse_regions(raw):
    """Validate the ``regions`` JSON field; returns
    ``[(id, (x, y, z), radius), ...]`` in request order."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ApiError("invalid_regions", "regions field is not valid JSON",
                       field="regions")
    if not isinstance(data, list):
        raise ApiError("invalid_regions",
                       "regions must be a JSON array of objects", field="regions")
    if not 1 <= len(data) <= MAX_REGIONS:
        raise ApiError("invalid_regions",
                       f"expected between 1 and {MAX_REGIONS} regions, got "
                       f"{len(data)}",
                       field="regions")
    seen = set()
    regions = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ApiError("invalid_regions",
                           f"region #{index} is not a JSON object",
                           field="regions")
        rid = item.get("id")
        if isinstance(rid, bool) or not isinstance(rid, int) \
                or not 0 <= rid < 2 ** 31:
            raise ApiError("invalid_regions",
                           f"region #{index} has invalid id {rid!r}; need an "
                           f"integer in [0, 2^31)",
                           field="regions")
        if rid in seen:
            raise ApiError("invalid_regions", f"duplicate region id {rid}",
                           field="regions")
        seen.add(rid)
        center = item.get("center")
        if not isinstance(center, list) or len(center) != 3:
            raise ApiError("invalid_regions",
                           f"region id {rid} needs \"center\": [x, y, z]",
                           field="regions")
        xyz = []
        for c in center:
            if isinstance(c, bool) or not isinstance(c, (int, float)) \
                    or not math.isfinite(c):
                raise ApiError("invalid_regions",
                               f"region id {rid} has a non-finite or "
                               f"non-numeric center coordinate {c!r}",
                               field="regions")
            xyz.append(float(c))
        radius = item.get("radius")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) \
                or not math.isfinite(radius) or radius <= 0:
            raise ApiError("invalid_regions",
                           f"region id {rid} has invalid radius {radius!r}; "
                           f"need a positive finite number",
                           field="regions")
        regions.append((rid, tuple(xyz), float(radius)))
    return regions


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "NiftiSampler/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------------
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def _send_json(self, status, obj, *, close=False):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def handle_expect_100(self):
        # Reject oversized uploads before the client streams the body.
        length = self.headers.get("Content-Length")
        try:
            n = int(length) if length is not None else 0
        except ValueError:
            n = 0
        if n > MAX_BODY_BYTES:
            self._send_json(413, {"error": {
                "code": "file_too_large",
                "message": f"request body of {n} bytes exceeds the "
                           f"{MAX_FILE_BYTES}-byte (16 MiB) file limit",
                "field": "file",
            }}, close=True)
            return False
        return super().handle_expect_100()

    # -- routes ---------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == HEALTH_PATH:
            if getattr(self.server, "ready", False):
                self._send_json(200, {"status": "ok", "ready": True})
            else:
                self._send_json(503, {"status": "unavailable", "ready": False})
        elif path in (SAMPLE_PATH, SPHERE_PATH):
            self._method_not_allowed()
        else:
            self._send_json(404, {"error": {"code": "not_found",
                                            "message": "unknown route"}})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == SAMPLE_PATH:
            handler = self._handle_sample
        elif path == SPHERE_PATH:
            handler = self._handle_sphere_stats
        else:
            self._send_json(404, {"error": {"code": "not_found",
                                            "message": "unknown route"}}, close=True)
            return
        try:
            handler()
        except ApiError as err:
            self._send_json(err.status, err.to_dict(), close=True)
        except Exception:  # pragma: no cover - defensive
            log.exception("unhandled error while serving %s", path)
            self._send_json(500, {"error": {"code": "internal_error",
                                            "message": "unexpected server error"}},
                            close=True)

    def do_PUT(self):
        self._method_not_allowed()

    def do_DELETE(self):
        self._method_not_allowed()

    def do_PATCH(self):
        self._method_not_allowed()

    def do_HEAD(self):
        self._method_not_allowed()

    def _method_not_allowed(self):
        self._send_json(405, {"error": {"code": "method_not_allowed",
                                        "message": "method not allowed"}})

    # -- endpoints ------------------------------------------------------------
    def _read_parts(self):
        """Read and parse one multipart/form-data request body."""
        ctype = self.headers.get("Content-Type", "")
        media, _, params = ctype.partition(";")
        if media.strip().lower() != "multipart/form-data":
            raise ApiError("invalid_multipart",
                           "Content-Type must be multipart/form-data", status=415)
        boundary = None
        for token in params.split(";"):
            key, sep, value = token.partition("=")
            if sep and key.strip().lower() == "boundary":
                boundary = value.strip().strip('"')
        if not boundary or len(boundary) > 200:
            raise ApiError("invalid_multipart", "missing or invalid multipart boundary")

        length = self.headers.get("Content-Length")
        if length is None:
            raise ApiError("missing_content_length",
                           "Content-Length header is required", status=411)
        try:
            n = int(length)
        except ValueError:
            raise ApiError("invalid_content_length",
                           f"invalid Content-Length {length!r}")
        if n < 0 or n > MAX_BODY_BYTES:
            raise ApiError("file_too_large",
                           f"request body of {n} bytes exceeds the "
                           f"{MAX_FILE_BYTES}-byte (16 MiB) file limit",
                           field="file", status=413)
        body = self.rfile.read(n)
        if len(body) != n:
            raise ApiError("invalid_multipart", "request body truncated")
        return parse_multipart(body, boundary)

    @staticmethod
    def _single_file(parts):
        files = [p for p in parts if p.filename is not None]
        if not files:
            raise ApiError("missing_file",
                           "multipart form must contain exactly one NIfTI file part",
                           field="file")
        if len(files) > 1:
            raise ApiError("multiple_files",
                           f"expected a single NIfTI file part, got {len(files)}",
                           field="file")
        content = files[0].content
        if len(content) > MAX_FILE_BYTES:
            raise ApiError("file_too_large",
                           f"NIfTI file is {len(content)} bytes; limit is "
                           f"{MAX_FILE_BYTES} (16 MiB)",
                           field="file", status=413)
        return content

    @staticmethod
    def _json_field(parts, name):
        fields = [p for p in parts if p.name == name and p.filename is None]
        if not fields:
            raise ApiError(f"missing_{name}",
                           f"multipart form must contain a {name!r} JSON field",
                           field=name)
        if len(fields) > 1:
            raise ApiError("invalid_multipart", f"multiple {name!r} fields")
        return fields[0].content

    def _handle_sample(self):
        parts = self._read_parts()
        content = self._single_file(parts)
        point_parts = [p for p in parts if p.name == "points"
                       and p.filename is None]
        if not point_parts:
            raise ApiError("missing_points",
                           "multipart form must contain a 'points' JSON field",
                           field="points")
        if len(point_parts) > 1:
            raise ApiError("invalid_multipart", "multiple 'points' fields")

        points = parse_points(point_parts[0].content)
        volume = parse_nifti(content)

        results = []
        for pid, coord in points:
            try:
                voxel, intensity = sample_point(volume, coord)
            except PointError as err:
                entry = {"id": pid, "status": "error",
                         "error": {"code": err.code, "message": err.message}}
                if err.voxel is not None:
                    entry["error"]["voxel"] = list(err.voxel)
                results.append(entry)
            else:
                results.append({"id": pid, "status": "ok",
                                "voxel": list(voxel), "intensity": intensity,
                                "transform": volume.transform})
        self._send_json(200, {"transform": volume.transform, "results": results})

    def _handle_sphere_stats(self):
        parts = self._read_parts()
        content = self._single_file(parts)
        regions = parse_regions(self._json_field(parts, "regions"))
        volume = parse_nifti(content)

        results = []
        for rid, center, radius in regions:
            try:
                stats = sphere_stats(volume, center, radius)
            except RegionError as err:
                results.append({"id": rid, "status": "error",
                                "error": {"code": err.code,
                                          "message": err.message}})
            else:
                results.append({"id": rid, "status": "ok", **stats})
        self._send_json(200, {"transform": volume.transform, "results": results})


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def self_check():
    """Exercise the sampler on a synthetic volume; readiness gate for /healthz."""
    ident = [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
    vol = NiftiVolume(dims=(2, 2, 2), datatype=4, endian="<", vox_offset=352,
                      scl_slope=1.0, scl_inter=0.0, transform="sform",
                      affine=ident, inverse=[row[:] for row in ident],
                      data=array("h", range(8)))
    (f, corner) = sample_point(vol, (1.0, 1.0, 1.0))
    if f != (1.0, 1.0, 1.0) or corner != 7.0:
        raise RuntimeError(f"self-check corner sample failed: {f} -> {corner}")
    (_, center) = sample_point(vol, (0.5, 0.5, 0.5))
    if abs(center - 3.5) > 1e-12:
        raise RuntimeError(f"self-check center sample failed: {center}")
    stats = sphere_stats(vol, (0.0, 0.0, 0.0), 1.0)
    # closed ball of radius 1 on the identity grid: origin plus the three
    # axis-neighbours sitting exactly on the sphere surface
    if stats["count"] != 4 or stats["min"] != 0.0 or stats["max"] != 4.0 \
            or abs(stats["mean"] - 1.75) > 1e-12:
        raise RuntimeError(f"self-check sphere stats failed: {stats}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        port = int(os.environ.get("PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        log.error("invalid PORT %r", os.environ.get("PORT"))
        return 2
    ready = True
    try:
        self_check()
    except Exception:
        ready = False
        log.exception("startup self-check failed; /healthz will report not-ready")
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    server.ready = ready
    log.info("listening on 0.0.0.0:%d (ready=%s)", port, ready)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
