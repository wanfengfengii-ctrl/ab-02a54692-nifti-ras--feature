"""Closed-ball adjudication and local-statistics tests for app.sphere."""
import math
import random
import unittest

from app.errors import RegionError
from app.nifti import parse_nifti
from app.sphere import DIST_ABS_TOL, DIST_EPS, MAX_SPHERE_VOXELS, sphere_stats
from verify.nifti_gen import build_nifti


def linear_fn(i, j, k):
    return i + 10 * j + 100 * k


def make_vol(dims=(6, 6, 6), data_fn=linear_fn, transform="sform", **kw):
    if transform == "sform":
        kw.setdefault("srow_x", (2.0, 0.0, 0.0, 10.0))
        kw.setdefault("srow_y", (0.0, 3.0, 0.0, 20.0))
        kw.setdefault("srow_z", (0.0, 0.0, 4.0, 30.0))
    else:
        kw.setdefault("quatern", (0.0, 0.0, math.sqrt(0.5)))
        kw.setdefault("qoffset", (10.0, 20.0, 30.0))
        kw.setdefault("pixdim", (1.0, 2.0, 3.0, 4.0))
    raw = build_nifti(datatype="float32", dims=dims, data_fn=data_fn,
                      transform=transform, **kw)
    return parse_nifti(raw)


def world_point(affine, i, j, k):
    return tuple(affine[r][0] * i + affine[r][1] * j + affine[r][2] * k
                 + affine[r][3] for r in range(3))


def identity_vol(dims=(4, 4, 4), data_fn=linear_fn, **kw):
    raw = build_nifti(datatype="float32", dims=dims, data_fn=data_fn,
                      srow_x=(1, 0, 0, 0), srow_y=(0, 1, 0, 0),
                      srow_z=(0, 0, 1, 0), **kw)
    return parse_nifti(raw)


def reference_stats(volume, center, radius):
    """Brute-force reference: every voxel center adjudicated in world space."""
    nx, ny, nz = volume.dims
    r2 = radius * radius
    vals = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                wx, wy, wz = world_point(volume.affine, i, j, k)
                dx, dy, dz = wx - center[0], wy - center[1], wz - center[2]
                if dx * dx + dy * dy + dz * dz <= r2:
                    vals.append(volume.value_at(i, j, k))
    return vals


class SphereGeometryTests(unittest.TestCase):
    def assert_matches_reference(self, volume, center, radius):
        stats = sphere_stats(volume, center, radius)
        vals = reference_stats(volume, center, radius)
        self.assertTrue(vals, "reference unexpectedly empty")
        self.assertEqual(stats["count"], len(vals))
        self.assertAlmostEqual(stats["min"], min(vals), places=6)
        self.assertAlmostEqual(stats["max"], max(vals), places=6)
        self.assertAlmostEqual(stats["mean"], sum(vals) / len(vals), places=6)
        return stats

    def assert_empty_like_reference(self, volume, center, radius):
        self.assertEqual(reference_stats(volume, center, radius), [])
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(volume, center, radius)
        self.assertEqual(ctx.exception.code, "empty_region")

    def test_single_center_ball(self):
        vol = make_vol(dims=(3, 3, 3))
        center = world_point(vol.affine, 1, 1, 1)
        # nearest neighbour is 2 mm away on x; 1.9 mm admits only the center
        stats = sphere_stats(vol, center, 1.9)
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["min"], stats["max"])
        self.assertEqual(stats["max"], stats["mean"])
        self.assertEqual(stats["min"], 111.0)

    def test_surface_is_closed(self):
        vol = identity_vol(dims=(5, 5, 5))
        center = (2.0, 2.0, 2.0)
        # radius exactly equal to the neighbour distance (1 mm): included
        stats = sphere_stats(vol, center, 1.0)
        self.assertEqual(stats["count"], 7)  # center + 6 face neighbours
        # slightly smaller radius: the six surface neighbours drop out
        stats_in = sphere_stats(vol, center, 1.0 - 1e-9)
        self.assertEqual(stats_in["count"], 1)

    def test_exact_surface_with_large_translation(self):
        # float32 header entries plus a large scanner offset: the neighbor is
        # exactly 1 mm from the center in world space and must still count.
        raw = build_nifti(datatype="float32", dims=(5, 5, 5), data_fn=linear_fn,
                          srow_x=(1.0, 0.0, 0.0, 1234.0),
                          srow_y=(0.0, 1.0, 0.0, -567.0),
                          srow_z=(0.0, 0.0, 1.0, 890.0))
        vol = parse_nifti(raw)
        center = world_point(vol.affine, 2, 2, 2)
        stats = sphere_stats(vol, center, 1.0)
        self.assertEqual(stats["count"], 7)

    def test_ball_encompassing_whole_volume(self):
        vol = make_vol(dims=(4, 5, 6))
        # center at volume center in world space; radius to the farthest
        # corner voxel center, computed from the same affine
        cc = world_point(vol.affine, 1.5, 2.0, 2.5)
        corners = [world_point(vol.affine, i, j, k)
                   for i in (0, 3) for j in (0, 4) for k in (0, 5)]
        radius = max(math.dist(cc, w) for w in corners)
        stats = sphere_stats(vol, cc, radius)
        self.assertEqual(stats["count"], 4 * 5 * 6)

    def test_stats_against_reference_sform(self):
        vol = make_vol(dims=(6, 7, 5))
        center = world_point(vol.affine, 2.3, 3.1, 1.7)
        self.assert_matches_reference(vol, center, 4.2)
        self.assert_matches_reference(vol, center, 2.0)
        # tiny ball around one exact voxel center admits just that center
        self.assert_matches_reference(
            vol, world_point(vol.affine, 1, 1, 1), 1.9)
        # ball genuinely missing every center is an empty region for both
        self.assert_empty_like_reference(vol, center, 0.5)

    def test_stats_against_reference_qform_rotation(self):
        vol = make_vol(transform="qform", dims=(6, 6, 6))
        center = world_point(vol.affine, 2.0, 3.0, 1.0)
        self.assert_matches_reference(vol, center, 5.5)
        self.assert_matches_reference(vol, center, 0.01)

    def test_anisotropic_world_grid(self):
        # 1 x 1 x 5 mm voxels: a 3 mm ball spans several x/y columns but a
        # single z plane.
        raw = build_nifti(datatype="float32", dims=(9, 9, 5), data_fn=linear_fn,
                          srow_x=(1.0, 0.0, 0.0, 0.0),
                          srow_y=(0.0, 1.0, 0.0, 0.0),
                          srow_z=(0.0, 0.0, 5.0, 0.0))
        vol = parse_nifti(raw)
        stats = sphere_stats(vol, (4.0, 4.0, 10.0), 3.0)
        # lattice points (i, j) with i^2 + j^2 <= 9 in the single k == 2 plane
        expected = sum(1 for i in range(-3, 4) for j in range(-3, 4)
                       if i * i + j * j <= 9)
        self.assertEqual(stats["count"], expected)
        self.assertEqual(expected, 29)

    def test_ball_extends_outside_image(self):
        vol = make_vol(dims=(6, 6, 6))
        center = world_point(vol.affine, 0, 0, 0)
        stats = self.assert_matches_reference(vol, center, 5.0)
        self.assertLess(stats["count"], 6 ** 3)

    def test_random_affines_match_reference(self):
        rng = random.Random(20261006)
        for trial in range(300):
            # random invertible affine: rotation-ish via QR-free random
            # symmetric-ish matrix + diagonal guaranteeing nonsingularity
            m = [[rng.uniform(-1.5, 1.5) for _ in range(3)] for _ in range(3)]
            for r in range(3):
                m[r][r] += rng.choice((-1.0, 1.0)) * rng.uniform(1.5, 3.0)
            det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                   - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            if abs(det) < 0.5:
                continue
            dims = (rng.randint(1, 7), rng.randint(1, 7), rng.randint(1, 7))
            t = (rng.uniform(-20.0, 20.0), rng.uniform(-20.0, 20.0),
                 rng.uniform(-20.0, 20.0))
            raw = build_nifti(
                datatype="float32", dims=dims, data_fn=linear_fn,
                srow_x=(m[0][0], m[0][1], m[0][2], t[0]),
                srow_y=(m[1][0], m[1][1], m[1][2], t[1]),
                srow_z=(m[2][0], m[2][1], m[2][2], t[2]))
            vol = parse_nifti(raw)
            # center near the image in world space, radius from a sliver to
            # several voxel spacings
            v0 = [rng.uniform(-1.5, d + 0.5) for d in dims]
            center = world_point(vol.affine, *v0)
            radius = rng.uniform(0.05, 6.0)
            vals = reference_stats(vol, center, radius)
            if not vals or len(vals) > MAX_SPHERE_VOXELS:
                with self.assertRaises(RegionError):
                    sphere_stats(vol, center, radius)
                continue
            stats = sphere_stats(vol, center, radius)
            self.assertEqual(stats["count"], len(vals),
                             msg=f"trial {trial}: {dims} {m} {center} {radius}")
            self.assertAlmostEqual(stats["min"], min(vals), places=5)
            self.assertAlmostEqual(stats["max"], max(vals), places=5)
            self.assertAlmostEqual(stats["mean"], sum(vals) / len(vals),
                                   places=5)


class SphereValueTests(unittest.TestCase):
    def test_min_max_mean_basic(self):
        raw = build_nifti(datatype="int16", dims=(3, 3, 3),
                          data_fn=lambda i, j, k: i + j + k,
                          srow_x=(1, 0, 0, 0), srow_y=(0, 1, 0, 0),
                          srow_z=(0, 0, 1, 0))
        vol = parse_nifti(raw)
        stats = sphere_stats(vol, (1.0, 1.0, 1.0), 1.0)
        self.assertEqual(stats["count"], 7)
        # values i+j+k: center=3; face neighbors 2,2,2,4,4,4
        self.assertEqual(stats["min"], 2.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertAlmostEqual(stats["mean"], 3.0)

    def test_scaling_applied(self):
        raw = build_nifti(datatype="int16", dims=(3, 3, 3),
                          data_fn=lambda i, j, k: 5, slope=2.0, inter=3.0,
                          srow_x=(1, 0, 0, 0), srow_y=(0, 1, 0, 0),
                          srow_z=(0, 0, 1, 0))
        vol = parse_nifti(raw)
        stats = sphere_stats(vol, (1.0, 1.0, 1.0), 0.5)
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["min"], 13.0)
        self.assertEqual(stats["max"], 13.0)
        self.assertEqual(stats["mean"], 13.0)

    def test_slope_zero_means_unscaled(self):
        raw = build_nifti(datatype="int16", dims=(3, 3, 3),
                          data_fn=lambda i, j, k: 7, slope=0.0, inter=999.0,
                          srow_x=(1, 0, 0, 0), srow_y=(0, 1, 0, 0),
                          srow_z=(0, 0, 1, 0))
        vol = parse_nifti(raw)
        stats = sphere_stats(vol, (1.0, 1.0, 1.0), 0.5)
        self.assertEqual(stats["min"], 7.0)


class SphereErrorTests(unittest.TestCase):
    def test_empty_far_outside(self):
        vol = identity_vol()
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (100.0, 100.0, 100.0), 1.0)
        self.assertEqual(ctx.exception.code, "empty_region")

    def test_empty_between_voxels(self):
        vol = identity_vol()
        # half a voxel from every center; radius < sqrt(3)/2 catches none
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (0.5, 0.5, 0.5), 0.8)
        self.assertEqual(ctx.exception.code, "empty_region")

    def test_non_finite_data(self):
        vol = identity_vol(data_fn=lambda i, j, k: float("nan")
                           if (i, j, k) == (1, 1, 1) else linear_fn(i, j, k))
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (1.0, 1.0, 1.0), 1.5)
        self.assertEqual(ctx.exception.code, "non_finite_data")
        self.assertEqual(ctx.exception.voxel, (1, 1, 1))

    def test_non_finite_far_voxel_not_adjudicated(self):
        # ball that excludes the bad voxel succeeds
        vol = identity_vol(data_fn=lambda i, j, k: float("inf")
                           if (i, j, k) == (2, 2, 2) else linear_fn(i, j, k))
        stats = sphere_stats(vol, (0.0, 0.0, 0.0), 1.0)
        self.assertEqual(stats["count"], 4)
        self.assertTrue(all(math.isfinite(stats[k])
                            for k in ("min", "max", "mean")))

    def test_capacity_boundary(self):
        # exactly 64*64*16 = 65536 voxel centers; radius through the corners
        nx, ny, nz = 64, 64, 16
        vol = identity_vol(dims=(nx, ny, nz), data_fn=lambda i, j, k: 1)
        c = ((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)
        hx, hy, hz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
        radius = math.sqrt(hx * hx + hy * hy + hz * hz)
        stats = sphere_stats(vol, c, radius)
        self.assertEqual(stats["count"], MAX_SPHERE_VOXELS)

    def test_capacity_exceeded(self):
        nx, ny, nz = 64, 64, 17  # 69632 voxels
        vol = identity_vol(dims=(nx, ny, nz), data_fn=lambda i, j, k: 1)
        c = ((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)
        hx, hy, hz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
        radius = math.sqrt(hx * hx + hy * hy + hz * hz)
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, c, radius)
        self.assertEqual(ctx.exception.code, "region_too_large")

    def test_capacity_fast_path_on_fine_grid(self):
        # 0.01 mm voxels on a 160^3 grid: a 0.8 mm ball physically encloses
        # ~2.1 million centers.  The O(1) inscribed-box bound (~149k) exceeds
        # the cap, so rejection happens before scanning million-strong boxes.
        step = 0.01
        n = 160
        raw = build_nifti(datatype="int16", dims=(n, n, n),
                          data_fn=lambda i, j, k: 1,
                          srow_x=(step, 0, 0, 0), srow_y=(0, step, 0, 0),
                          srow_z=(0, 0, step, 0))
        vol = parse_nifti(raw)
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (0.8, 0.8, 0.8), 0.8)
        self.assertEqual(ctx.exception.code, "region_too_large")

    def test_extreme_finite_inputs_do_not_raise(self):
        vol = identity_vol()
        # Huge center with a huge finite radius that still misses the tiny
        # image; and a huge radius that sweeps the whole image.
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (1.0e308, 0.0, 0.0), 1.0)
        self.assertEqual(ctx.exception.code, "empty_region")
        stats = sphere_stats(vol, (1.0e308, 0.0, 0.0), 2.0e308)
        self.assertEqual(stats["count"], 4 ** 3)
        # Tiny affine near the float32 floor vs. extreme centers
        tiny = parse_nifti(build_nifti(
            datatype="float32", dims=(2, 2, 2), data_fn=linear_fn,
            srow_x=(1.0e-30, 0.0, 0.0, 0.0),
            srow_y=(0.0, 1.0e-30, 0.0, 0.0),
            srow_z=(0.0, 0.0, 1.0e-30, 0.0)))
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(tiny, (1.0e308, 0.0, 0.0), 1.0)
        self.assertEqual(ctx.exception.code, "empty_region")


if __name__ == "__main__":
    unittest.main()
