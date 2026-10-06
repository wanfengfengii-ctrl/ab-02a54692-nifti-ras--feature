"""End-to-end tests for POST /api/nifti/sphere-stats."""
import http.client
import json
import logging
import math
import threading
import unittest
from http.server import ThreadingHTTPServer

from app.server import Handler, MAX_FILE_BYTES
from verify.httpclient import BOUNDARY, get_json, post_sphere_stats
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


class SphereApiTests(unittest.TestCase):
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

    # -- happy paths -----------------------------------------------------------
    def test_single_voxel_center_ball(self):
        regions = [{"id": 3, "center": world_of((1, 1, 1)), "radius": 1.9}]
        status, payload = post_sphere_stats(self.base, sform_file(), regions)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["transform"], "sform")
        (r,) = payload["results"]
        self.assertEqual(r["id"], 3)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["count"], 1)
        self.assertEqual(r["min"], 111.0)
        self.assertEqual(r["max"], 111.0)
        self.assertEqual(r["mean"], 111.0)
        self.assertEqual(r["transform"], "sform")

    def test_closed_surface_and_order(self):
        # radius exactly 2 mm: the +/- 2 mm x neighbors sit on the surface
        regions = [
            {"id": 9, "center": world_of((2, 2, 2)), "radius": 2.0},
            {"id": 1, "center": world_of((0, 0, 0)), "radius": 1.9},
        ]
        status, payload = post_sphere_stats(self.base, sform_file(), regions)
        self.assertEqual(status, 200, payload)
        self.assertEqual([r["id"] for r in payload["results"]], [9, 1])
        r9, r1 = payload["results"]
        self.assertEqual(r9["count"], 3)  # center + 2 surface neighbors
        self.assertEqual(r1["count"], 1)

    def test_stats_values(self):
        # identity grid, values i+j+k around center (1,1,1), r=1
        raw = build_nifti(datatype="int16", dims=(3, 3, 3),
                          data_fn=lambda i, j, k: i + j + k,
                          srow_x=(1, 0, 0, 0), srow_y=(0, 1, 0, 0),
                          srow_z=(0, 0, 1, 0))
        regions = [{"id": 1, "center": [1.0, 1.0, 1.0], "radius": 1.0}]
        status, payload = post_sphere_stats(self.base, raw, regions)
        self.assertEqual(status, 200, payload)
        r = payload["results"][0]
        self.assertEqual(r["count"], 7)
        self.assertEqual(r["min"], 2.0)
        self.assertEqual(r["max"], 4.0)
        self.assertAlmostEqual(r["mean"], 3.0)

    def test_big_endian_qform(self):
        quat = (0.0, 0.0, math.sqrt(0.5))  # rotz(+90 deg)
        raw = build_nifti(endian=">", datatype="float32", dims=(4, 5, 6),
                          data_fn=data_fn, transform="qform",
                          quatern=quat, qoffset=(10.0, 20.0, 30.0),
                          pixdim=(1.0, 2.0, 3.0, 4.0))
        # affine [[0,-3,0,10],[2,0,0,20],[0,0,4,30]]; ball of radius 0.1 mm
        # around voxel (1,2,3) admits only that center
        world = [0.0 * 1 - 3.0 * 2 + 10.0, 2.0 * 1 + 20.0, 4.0 * 3 + 30.0]
        status, payload = post_sphere_stats(
            self.base, raw, [{"id": 1, "center": world, "radius": 0.1}])
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["transform"], "qform")
        r = payload["results"][0]
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["count"], 1)
        self.assertAlmostEqual(r["mean"], data_fn(1, 2, 3), places=4)
        self.assertEqual(r["transform"], "qform")

    def test_scaling_applied_to_stats(self):
        raw = sform_file(slope=2.0, inter=5.0)
        regions = [{"id": 1, "center": world_of((1, 1, 1)), "radius": 1.9}]
        status, payload = post_sphere_stats(self.base, raw, regions)
        self.assertEqual(status, 200, payload)
        r = payload["results"][0]
        self.assertEqual(r["min"], 111.0 * 2.0 + 5.0)
        self.assertEqual(r["mean"], 111.0 * 2.0 + 5.0)

    # -- per-region failures ---------------------------------------------------
    def test_empty_region_is_per_region_error(self):
        regions = [
            {"id": 1, "center": world_of((1, 1, 1)), "radius": 1.9},
            {"id": 2, "center": [1.0e5, 1.0e5, 1.0e5], "radius": 1.0},
            {"id": 3, "center": world_of((3, 4, 5)), "radius": 1.9},
        ]
        status, payload = post_sphere_stats(self.base, sform_file(), regions)
        self.assertEqual(status, 200, payload)
        r1, r2, r3 = payload["results"]
        self.assertEqual(r1["status"], "ok")
        self.assertEqual(r2["status"], "error")
        self.assertEqual(r2["error"]["code"], "empty_region")
        self.assertEqual(r3["status"], "ok")
        self.assertEqual(r3["count"], 1)

    def test_non_finite_data_is_per_region_error(self):
        raw = sform_file(datatype="float32",
                         data_fn=lambda i, j, k: float("nan")
                         if (i, j, k) == (1, 1, 1) else data_fn(i, j, k))
        regions = [
            {"id": 7, "center": world_of((1, 1, 1)), "radius": 1.0},
            {"id": 8, "center": world_of((3, 4, 5)), "radius": 1.0},
        ]
        status, payload = post_sphere_stats(self.base, raw, regions)
        self.assertEqual(status, 200, payload)
        r7, r8 = payload["results"]
        self.assertEqual(r7["status"], "error")
        self.assertEqual(r7["error"]["code"], "non_finite_data")
        self.assertEqual(r7["error"]["voxel"], [1, 1, 1])
        self.assertEqual(r8["status"], "ok")

    def test_region_too_large(self):
        # identity volume of 41^3 = 68921 voxels, ball through all corners
        n = 41
        raw = build_nifti(datatype="int16", dims=(n, n, n),
                          data_fn=lambda i, j, k: 1,
                          srow_x=(1, 0, 0, 0), srow_y=(0, 1, 0, 0),
                          srow_z=(0, 0, 1, 0))
        h = (n - 1) / 2.0
        radius = math.sqrt(3.0) * h
        regions = [
            {"id": 1, "center": [h, h, h], "radius": radius},
            {"id": 2, "center": [0.0, 0.0, 0.0], "radius": 0.5},
        ]
        status, payload = post_sphere_stats(self.base, raw, regions)
        self.assertEqual(status, 200, payload)
        r1, r2 = payload["results"]
        self.assertEqual(r1["status"], "error")
        self.assertEqual(r1["error"]["code"], "region_too_large")
        self.assertEqual(r2["status"], "ok")
        self.assertEqual(r2["count"], 1)

    # -- regions validation ----------------------------------------------------
    def assert_request_error(self, status, payload, code, field="regions"):
        self.assertEqual(status, 400, payload)
        self.assertEqual(payload["error"]["code"], code)
        self.assertEqual(payload["error"]["field"], field)

    def test_invalid_regions(self):
        raw = sform_file()
        cases = [
            [],
            [{"id": i, "center": [0, 0, 0], "radius": 1.0} for i in range(33)],
            [{"id": 1, "center": [0, 0, 0], "radius": 1.0},
             {"id": 1, "center": [1, 1, 1], "radius": 1.0}],
            [{"id": 1, "center": [float("nan"), 0, 0], "radius": 1.0}],
            [{"id": -1, "center": [0, 0, 0], "radius": 1.0}],
            [{"id": 1, "center": [0, 0], "radius": 1.0}],
            [{"id": 1, "radius": 1.0}],
            [{"id": 1, "center": [0, 0, 0], "radius": 0.0}],
            [{"id": 1, "center": [0, 0, 0], "radius": -1.0}],
            [{"id": 1, "center": [0, 0, 0], "radius": float("inf")}],
            [{"id": 1, "center": [0, 0, 0]}],
            [{"id": 1}],
            b"[{",
            b"{}",
        ]
        for regions in cases:
            with self.subTest(regions=str(regions)[:60]):
                status, payload = post_sphere_stats(self.base, raw, regions)
                self.assert_request_error(status, payload, "invalid_regions")

    def test_missing_regions_field(self):
        body = (b"--" + BOUNDARY.encode() + b"\r\n"
                b'Content-Disposition: form-data; name="file"; filename="v.nii"\r\n'
                b"Content-Type: application/octet-stream\r\n\r\n" + sform_file() +
                b"\r\n--" + BOUNDARY.encode() + b"--\r\n")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "missing_regions")
        self.assertEqual(payload["error"]["field"], "regions")

    # -- routing / compatibility ------------------------------------------------
    def test_sample_still_works_with_regions_endpoint_present(self):
        from verify.httpclient import post_sample
        status, payload = post_sample(
            self.base, sform_file(),
            [{"id": 1, "point": world_of((1, 1, 1))}])
        self.assertEqual(status, 200)
        self.assertEqual(payload["results"][0]["intensity"], 111.0)

    def test_get_on_sphere_path_is_405(self):
        status, payload = get_json(self.base, "/api/nifti/sphere-stats")
        self.assertEqual(status, 405)
        self.assertEqual(payload["error"]["code"], "method_not_allowed")

    def test_points_payload_rejected_on_sphere_endpoint(self):
        from verify.httpclient import build_field_multipart
        body = build_field_multipart(sform_file(),
                                     [{"id": 1, "point": [0, 0, 0]}],
                                     field_name="points")
        status, payload = self.post_raw(
            body, f"multipart/form-data; boundary={BOUNDARY}")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "missing_regions")

    def test_file_too_large(self):
        raw = b"\x00" * (MAX_FILE_BYTES + 1)
        status, payload = post_sphere_stats(
            self.base, raw, [{"id": 1, "center": [0, 0, 0], "radius": 1.0}])
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "file_too_large")


if __name__ == "__main__":
    unittest.main()
