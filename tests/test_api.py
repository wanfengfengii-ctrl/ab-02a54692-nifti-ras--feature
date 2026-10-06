"""End-to-end API tests against a live in-process HTTP server."""
import http.client
import json
import logging
import math
import threading
import unittest
from http.server import ThreadingHTTPServer

from app.server import Handler, MAX_FILE_BYTES
from verify.httpclient import BOUNDARY, build_multipart, get_json, post_sample
from verify.httpclient import post_sphere_stats
from verify.nifti_gen import build_nifti

SFORM = ((2.0, 0.0, 0.0, 10.0), (0.0, 3.0, 0.0, 20.0), (0.0, 0.0, 4.0, 30.0))


def data_fn(i, j, k):
    return i + 10 * j + 100 * k


def world_of(voxel):
    i, j, k = voxel
    return [2.0 * i + 10.0, 3.0 * j + 20.0, 4.0 * k + 30.0]


def sform_file(**overrides):
    kw = dict(endian="<", datatype="int16", dims=(4, 5, 6), data_fn=data_fn,
              transform="sform",
              srow_x=SFORM[0], srow_y=SFORM[1], srow_z=SFORM[2])
    kw.update(overrides)
    return build_nifti(**kw)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        cls.server.ready = True
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        logging.disable(logging.NOTSET)

    def post_raw(self, body, content_type):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("POST", "/api/nifti/sample", body=body,
                         headers={"Content-Type": content_type})
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        finally:
            conn.close()
        try:
            return status, json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            return status, None

    # -- happy paths ---------------------------------------------------------
    def test_sform_sampling_and_order(self):
        points = [
            {"id": 7, "point": world_of((1.0, 1.0, 1.0))},
            {"id": 3, "point": world_of((0.5, 0.5, 0.5))},
            {"id": 9, "point": world_of((3.0, 4.0, 5.0))},
        ]
        status, payload = post_sample(self.base, sform_file(), points)
        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"], "sform")
        self.assertEqual([r["id"] for r in payload["results"]], [7, 3, 9])
        r7, r3, r9 = payload["results"]
        self.assertEqual(r7["status"], "ok")
        self.assertEqual(r7["voxel"], [1.0, 1.0, 1.0])
        self.assertEqual(r7["intensity"], 111.0)
        self.assertEqual(r7["transform"], "sform")
        self.assertEqual(r3["status"], "ok")
        self.assertAlmostEqual(r3["intensity"], 55.5)
        self.assertEqual(r9["status"], "ok")
        self.assertEqual(r9["intensity"], 543.0)

    def test_big_endian_float32_qform(self):
        quat = (0.0, 0.0, math.sqrt(0.5))  # rotz(+90 deg)
        raw = build_nifti(endian=">", datatype="float32", dims=(4, 5, 6),
                          data_fn=data_fn, transform="qform",
                          quatern=quat, qoffset=(10.0, 20.0, 30.0),
                          pixdim=(1.0, 2.0, 3.0, 4.0))
        # intended affine: [[0,-3,0,10],[2,0,0,20],[0,0,4,30]]
        world = [0.0 * 1 - 3.0 * 2 + 10.0, 2.0 * 1 + 20.0, 4.0 * 3 + 30.0]
        status, payload = post_sample(self.base, raw,
                                      [{"id": 1, "point": world}])
        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"], "qform")
        (result,) = payload["results"]
        self.assertEqual(result["status"], "ok")
        for got, want in zip(result["voxel"], (1.0, 2.0, 3.0)):
            self.assertAlmostEqual(got, want, places=4)
        self.assertAlmostEqual(result["intensity"], 321.0, places=3)

    def test_per_point_errors_do_not_fail_request(self):
        points = [
            {"id": 1, "point": world_of((1.0, 1.0, 1.0))},
            {"id": 2, "point": world_of((-1.0, 0.0, 0.0))},
            {"id": 3, "point": world_of((0.0, 0.0, 6.0))},
        ]
        status, payload = post_sample(self.base, sform_file(), points)
        self.assertEqual(status, 200)
        ok, oob1, oob2 = payload["results"]
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(oob1["status"], "error")
        self.assertEqual(oob1["error"]["code"], "out_of_bounds")
        self.assertIn("voxel", oob1["error"])
        self.assertEqual(oob2["error"]["code"], "out_of_bounds")

    def test_non_finite_data_per_point(self):
        raw = sform_file(datatype="float32",
                         data_fn=lambda i, j, k: float("nan")
                         if (i, j, k) == (1, 1, 1) else data_fn(i, j, k))
        points = [{"id": 5, "point": world_of((0.5, 0.5, 0.5))}]
        status, payload = post_sample(self.base, raw, points)
        self.assertEqual(status, 200)
        (result,) = payload["results"]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "non_finite_data")

    # -- structural errors -----------------------------------------------------
    def assert_error(self, status, payload, want_status, code, field=None):
        self.assertEqual(status, want_status, payload)
        err = payload["error"]
        self.assertEqual(err["code"], code)
        if field is not None:
            self.assertEqual(err["field"], field)

    def test_structural_errors(self):
        cases = [
            (sform_file(extra=b"\x00"), "trailing_bytes", "file"),
            (sform_file(truncate=4), "payload_length_mismatch", "file"),
            (sform_file(qform_code=0, sform_code=0), "missing_affine",
             "qform_code,sform_code"),
            (sform_file(srow_x=(0, 0, 0, 0)), "singular_affine", "srow"),
            (sform_file(datatype_code=8), "unsupported_datatype", "datatype"),
            (sform_file(slope=float("nan")), "non_finite_scaling", "scl_slope"),
            (sform_file(dim0=4), "invalid_dimensions", "dim"),
            (sform_file(magic=b"ni1\x00"), "unsupported_magic", "magic"),
            (sform_file(vox_offset=100.0), "invalid_vox_offset", "vox_offset"),
            (sform_file(bitpix=8), "bitpix_mismatch", "bitpix"),
            (b"\x00" * 10, "header_too_short", "file"),
        ]
        for raw, code, field in cases:
            with self.subTest(code=code):
                status, payload = post_sample(self.base, raw,
                                              [{"id": 1, "point": [0, 0, 0]}])
                self.assert_error(status, payload, 400, code, field)

    def test_big_endian_structural_error(self):
        raw = build_nifti(endian=">", datatype="float32", dims=(2, 2, 2),
                          data_fn=data_fn, extra=b"zz")
        status, payload = post_sample(self.base, raw,
                                      [{"id": 1, "point": [0, 0, 0]}])
        self.assert_error(status, payload, 400, "trailing_bytes", "file")

    # -- points validation -----------------------------------------------------
    def test_invalid_points(self):
        raw = sform_file()
        cases = [
            [],
            [{"id": i, "point": [0, 0, 0]} for i in range(257)],
            [{"id": 1, "point": [0, 0, 0]}, {"id": 1, "point": [1, 1, 1]}],
            [{"id": 1, "point": [float("nan"), 0, 0]}],
            [{"id": -1, "point": [0, 0, 0]}],
            [{"id": 1, "point": [0, 0]}],
            [{"id": 1}],
            b"[{",
            b"{}",
        ]
        for points in cases:
            with self.subTest(points=str(points)[:60]):
                status, payload = post_sample(self.base, raw, points)
                self.assert_error(status, payload, 400, "invalid_points", "points")

    # -- form-level errors -----------------------------------------------------
    def test_missing_file_part(self):
        body = (b"--" + BOUNDARY.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="points"\r\n\r\n'
                b'[{"id": 1, "point": [0, 0, 0]}]\r\n'
                b"--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "missing_file", "file")

    def test_multiple_file_parts(self):
        raw = sform_file()
        body = build_multipart(raw, [{"id": 1, "point": [0, 0, 0]}])
        # inject a second file part before the closing boundary
        extra = (b"--" + BOUNDARY.encode() + b"\r\n"
                 b'Content-Disposition: form-data; name="f2"; filename="b.nii"\r\n'
                 b"Content-Type: application/octet-stream\r\n\r\n" + raw + b"\r\n")
        body = body.replace(b"--" + BOUNDARY.encode() + b"--\r\n",
                            extra + b"--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "multiple_files", "file")

    def test_missing_points_field(self):
        body = (b"--" + BOUNDARY.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="v.nii"\r\n'
                b"Content-Type: application/octet-stream\r\n\r\n" + sform_file() +
                b"\r\n--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assert_error(status, payload, 400, "missing_points", "points")

    def test_wrong_content_type(self):
        status, payload = self.post_raw(b"hello", "text/plain")
        self.assert_error(status, payload, 415, "invalid_multipart")

    def test_file_too_large(self):
        raw = b"\x00" * (MAX_FILE_BYTES + 1)
        status, payload = post_sample(self.base, raw,
                                      [{"id": 1, "point": [0, 0, 0]}])
        self.assert_error(status, payload, 413, "file_too_large", "file")

    # -- routing ----------------------------------------------------------------
    def test_healthz(self):
        status, payload = get_json(self.base, "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "ready": True})

    def test_unknown_routes(self):
        status, _ = get_json(self.base, "/nope")
        self.assertEqual(status, 404)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("POST", "/nope", body=b"{}")
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 404)
        finally:
            conn.close()

    def test_get_on_sample_path_is_405(self):
        status, payload = get_json(self.base, "/api/nifti/sample")
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")


SPHERE_SFORM = ((2.0, 0.0, 0.0, 10.0),
                (0.0, 3.0, 0.0, 20.0),
                (0.0, 0.0, 4.0, 30.0))


def sphere_file(**overrides):
    kw = dict(endian="<", datatype="int16", dims=(6, 6, 6), data_fn=data_fn,
              transform="sform",
              srow_x=SPHERE_SFORM[0], srow_y=SPHERE_SFORM[1],
              srow_z=SPHERE_SFORM[2])
    kw.update(overrides)
    return build_nifti(**kw)


def sphere_world(voxel):
    i, j, k = voxel
    return [2.0 * i + 10.0, 3.0 * j + 20.0, 4.0 * k + 30.0]


def expected_sphere_values(center_voxel, radius_mm):
    """Closed-ball reference on the sform grid (voxel spacing 2/3/4 mm)."""
    fi, fj, fk = center_voxel
    rx = radius_mm / 2.0
    ry = radius_mm / 3.0
    rz = radius_mm / 4.0
    vals = []
    for k in range(6):
        for j in range(6):
            for i in range(6):
                if ((i - fi) / rx) ** 2 + ((j - fj) / ry) ** 2 \
                        + ((k - fk) / rz) ** 2 <= 1.0 + 1e-12:
                    vals.append(data_fn(i, j, k))
    return vals


class SphereStatsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        cls.server.ready = True
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        logging.disable(logging.NOTSET)

    def post_raw(self, body, content_type, path="/api/nifti/sphere-stats"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("POST", path, body=body,
                         headers={"Content-Type": content_type})
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        finally:
            conn.close()
        try:
            return status, json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            return status, None

    def test_stats_and_order(self):
        regions = [
            {"id": 9, "center": sphere_world((2.0, 2.0, 2.0)), "radius": 3.0},
            {"id": 1, "center": sphere_world((0.0, 0.0, 0.0)), "radius": 0.5},
            {"id": 5, "center": sphere_world((3.0, 3.0, 3.0)), "radius": 2.5},
        ]
        status, payload = post_sphere_stats(self.base, sphere_file(), regions)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["transform"], "sform")
        self.assertEqual([r["id"] for r in payload["results"]], [9, 1, 5])
        r9, r1, r5 = payload["results"]
        for r in (r9, r1, r5):
            self.assertEqual(r["status"], "ok", r)
            self.assertEqual(r["transform"], "sform")
        self.assertEqual(r1["count"], 1)
        self.assertEqual(r1["min"], 0.0)
        self.assertEqual(r1["max"], 0.0)
        self.assertEqual(r1["mean"], 0.0)
        for r, (cv, rad) in ((r9, ((2.0, 2.0, 2.0), 3.0)),
                             (r5, ((3.0, 3.0, 3.0), 2.5))):
            vals = expected_sphere_values(cv, rad)
            self.assertEqual(r["count"], len(vals), r)
            self.assertAlmostEqual(r["min"], min(vals), places=4)
            self.assertAlmostEqual(r["max"], max(vals), places=4)
            self.assertAlmostEqual(r["mean"], sum(vals) / len(vals), places=4)

    def test_closed_ball_surface(self):
        # radius 2 mm around voxel (1,1,1): centres exactly 2 mm away
        # (i-1 = +/-1, same j,k) lie on the surface and must be included
        regions = [{"id": 1, "center": sphere_world((1.0, 1.0, 1.0)),
                    "radius": 2.0}]
        status, payload = post_sphere_stats(self.base, sphere_file(), regions)
        self.assertEqual(status, 200)
        (r,) = payload["results"]
        self.assertEqual(r["status"], "ok")
        vals = expected_sphere_values((1.0, 1.0, 1.0), 2.0)
        self.assertEqual(r["count"], len(vals))

    def test_sphere_outside_image_is_empty_error(self):
        regions = [
            {"id": 3, "center": sphere_world((2.0, 2.0, 2.0)), "radius": 1.0},
            {"id": 4, "center": [1000.0, 1000.0, 1000.0], "radius": 1.0},
            {"id": 5, "center": sphere_world((0.0, 0.0, 0.0)), "radius": 0.5},
        ]
        status, payload = post_sphere_stats(self.base, sphere_file(), regions)
        self.assertEqual(status, 200)
        r3, r4, r5 = payload["results"]
        self.assertEqual(r3["status"], "ok")
        self.assertEqual(r4["status"], "error")
        self.assertEqual(r4["id"], 4)
        self.assertEqual(r4["error"]["code"], "empty_region")
        self.assertEqual(r5["status"], "ok")

    def test_non_finite_intensity_region_error_is_isolated(self):
        raw = sphere_file(
            datatype="float32",
            data_fn=lambda i, j, k: float("nan") if (i, j, k) == (1, 1, 1)
            else data_fn(i, j, k))
        regions = [
            {"id": 1, "center": sphere_world((1.0, 1.0, 1.0)), "radius": 0.5},
            {"id": 2, "center": sphere_world((1.0, 1.0, 1.0)), "radius": 1.0},
            {"id": 3, "center": sphere_world((4.0, 4.0, 4.0)), "radius": 0.5},
        ]
        status, payload = post_sphere_stats(self.base, raw, regions)
        self.assertEqual(status, 200)
        r1, r2, r3 = payload["results"]
        self.assertEqual(r1["status"], "error")
        self.assertEqual(r1["error"]["code"], "non_finite_intensity")
        self.assertEqual(r2["status"], "error")
        self.assertEqual(r2["error"]["code"], "non_finite_intensity")
        self.assertEqual(r3["status"], "ok")
        self.assertEqual(r3["count"], 1)

    def test_capacity_exceeded_region_error(self):
        # 6x6x6 image = 216 voxels, far below the 65536 cap; force the cap
        # by monkeypatching the module constant used by the server path.
        from app import spheres as spheres_mod
        saved = spheres_mod.MAX_SPHERE_VOXELS
        spheres_mod.MAX_SPHERE_VOXELS = 8
        try:
            regions = [
                {"id": 1, "center": sphere_world((0.0, 0.0, 0.0)),
                 "radius": 0.5},
                {"id": 2, "center": sphere_world((2.0, 2.0, 2.0)),
                 "radius": 100.0},
                {"id": 3, "center": sphere_world((5.0, 5.0, 5.0)),
                 "radius": 0.5},
            ]
            status, payload = post_sphere_stats(self.base, sphere_file(),
                                                regions)
        finally:
            spheres_mod.MAX_SPHERE_VOXELS = saved
        self.assertEqual(status, 200)
        r1, r2, r3 = payload["results"]
        self.assertEqual(r1["status"], "ok")
        self.assertEqual(r2["status"], "error")
        self.assertEqual(r2["error"]["code"], "region_capacity_exceeded")
        self.assertEqual(r3["status"], "ok")

    def test_big_endian_qform_stats(self):
        quat = (0.0, 0.0, math.sqrt(0.5))  # rotz(+90 deg)
        raw = build_nifti(endian=">", datatype="float32", dims=(6, 6, 6),
                          data_fn=data_fn, transform="qform",
                          quatern=quat, qoffset=(10.0, 20.0, 30.0),
                          pixdim=(1.0, 2.0, 3.0, 4.0))
        regions = [{"id": 1, "center": [10.0, 22.0, 34.0], "radius": 0.5}]
        status, payload = post_sphere_stats(self.base, raw, regions)
        self.assertEqual(status, 200)
        self.assertEqual(payload["transform"], "qform")
        (r,) = payload["results"]
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["count"], 1)
        # center maps to voxel (1,0,1) under
        # [[0,-3,0,10],[2,0,0,20],[0,0,4,30]]
        self.assertAlmostEqual(r["mean"], 101.0, places=3)
        self.assertEqual(r["transform"], "qform")

    def test_invalid_regions(self):
        raw = sphere_file()
        cases = [
            [],
            [{"id": i, "center": [0, 0, 0], "radius": 1.0}
             for i in range(33)],
            [{"id": 1, "center": [0, 0, 0], "radius": 1.0},
             {"id": 1, "center": [1, 1, 1], "radius": 1.0}],
            [{"id": 1, "center": [0, 0], "radius": 1.0}],
            [{"id": 1, "center": [float("nan"), 0, 0], "radius": 1.0}],
            [{"id": 1, "center": [0, 0, 0], "radius": 0.0}],
            [{"id": 1, "center": [0, 0, 0], "radius": -1.0}],
            [{"id": 1, "center": [0, 0, 0], "radius": float("inf")}],
            [{"id": 1, "center": [0, 0, 0]}],
            [{"id": -1, "center": [0, 0, 0], "radius": 1.0}],
            b"[{",
            b"{}",
        ]
        for regions in cases:
            with self.subTest(regions=str(regions)[:60]):
                status, payload = post_sphere_stats(self.base, raw, regions)
                self.assertEqual(status, 400, payload)
                self.assertEqual(payload["error"]["code"], "invalid_regions")
                self.assertEqual(payload["error"]["field"], "regions")

    def test_form_and_routing_errors(self):
        # missing regions field
        body = (b"--" + BOUNDARY.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="v.nii"\r\n'
                b"Content-Type: application/octet-stream\r\n\r\n"
                + sphere_file()
                + b"\r\n--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "missing_regions")
        self.assertEqual(payload["error"]["field"], "regions")

        # wrong content type
        status, payload = self.post_raw(b"hello", "text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(payload["error"]["code"], "invalid_multipart")

        # GET on the sphere path is 405
        status, payload = get_json(self.base, "/api/nifti/sphere-stats")
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")

        # sample endpoint does not accept regions
        body = build_multipart(sphere_file(),
                               [{"id": 1, "center": [0, 0, 0], "radius": 1.0}],
                               payload_field="regions")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("POST", "/api/nifti/sample", body=body,
                         headers={"Content-Type":
                                  f"multipart/form-data; boundary={BOUNDARY}"})
            resp = conn.getresponse()
            status = resp.status
            payload = json.loads(resp.read())
        finally:
            conn.close()
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "missing_points")


if __name__ == "__main__":
    unittest.main()
