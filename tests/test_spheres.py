"""Closed-ball world-space statistics tests for app.spheres."""
import math
import unittest

from app.errors import RegionError
from app.nifti import parse_nifti
from app.spheres import MAX_SPHERE_VOXELS, sphere_stats
from verify.nifti_gen import build_nifti


def linear_fn(i, j, k):
    return i + 10 * j + 100 * k


def make_vol(dims=(5, 5, 5), data_fn=linear_fn, **kw):
    kw.setdefault("datatype", "float32")
    return parse_nifti(build_nifti(dims=dims, data_fn=data_fn, **kw))


def reference_stats(vol, center, radius):
    """Brute-force closed-ball reference using the parsed affine directly."""
    nx, ny, nz = vol.dims
    x, y, z = center
    values = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                wx = vol.affine[0][0] * i + vol.affine[0][1] * j \
                    + vol.affine[0][2] * k + vol.affine[0][3]
                wy = vol.affine[1][0] * i + vol.affine[1][1] * j \
                    + vol.affine[1][2] * k + vol.affine[1][3]
                wz = vol.affine[2][0] * i + vol.affine[2][1] * j \
                    + vol.affine[2][2] * k + vol.affine[2][3]
                d2 = (wx - x) ** 2 + (wy - y) ** 2 + (wz - z) ** 2
                if d2 <= radius * radius * (1 + 1e-12) + 1e-12:
                    values.append(vol.value_at(i, j, k))
    return values


class SphereStatsTests(unittest.TestCase):
    def assert_matches_reference(self, vol, center, radius):
        stats = sphere_stats(vol, center, radius)
        values = reference_stats(vol, center, radius)
        self.assertTrue(values)
        self.assertEqual(stats["count"], len(values))
        self.assertAlmostEqual(stats["min"], min(values), places=5)
        self.assertAlmostEqual(stats["max"], max(values), places=5)
        self.assertAlmostEqual(stats["mean"], math.fsum(values) / len(values),
                               places=5)
        self.assertEqual(stats["transform"], vol.transform)

    def test_single_voxel_radius_zero(self):
        vol = make_vol()
        stats = sphere_stats(vol, (1.0, 1.0, 1.0), 0.0)
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["min"], 111.0)
        self.assertEqual(stats["max"], 111.0)
        self.assertEqual(stats["mean"], 111.0)

    def test_closed_surface_centers_included(self):
        # identity grid, radius 1 around the origin voxel: voxel centres at
        # axis distance exactly 1 sit on the sphere and must count
        vol = make_vol()
        stats = sphere_stats(vol, (0.0, 0.0, 0.0), 1.0)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertAlmostEqual(stats["mean"], (0 + 1 + 10 + 100) / 4.0)

    def test_surface_just_inside_excludes_outer_shell(self):
        vol = make_vol()
        # an unmistakable (relative to the 1e-12 surface slack) step inside
        inner = sphere_stats(vol, (0.0, 0.0, 0.0), 1.0 - 1e-9)
        self.assertEqual(inner["count"], 1)

    def test_ball_is_in_world_not_voxel_space(self):
        vol = make_vol(srow_x=(2, 0, 0, 0), srow_y=(0, 2, 0, 0),
                       srow_z=(0, 0, 2, 0))
        # radius 2 mm reaches exactly one voxel (2 mm) out on every axis;
        # those six centres sit on the surface and must count -> 7 total
        stats = sphere_stats(vol, (4.0, 4.0, 4.0), 2.0)
        self.assertEqual(stats["count"], 7)
        # a clear step inside the surface: only the centre voxel remains
        tight = sphere_stats(vol, (4.0, 4.0, 4.0), 2.0 - 1e-9)
        self.assertEqual(tight["count"], 1)
        # 3.5 mm spans face/space diagonals of the 2 mm cube (max 3.464 mm)
        # but not the 4 mm two-step neighbours -> exactly a 3x3x3 block
        block = sphere_stats(vol, (4.0, 4.0, 4.0), 3.5)
        self.assertEqual(block["count"], 27)

    def test_anisotropic_affine_matches_reference(self):
        vol = make_vol(
            dims=(6, 7, 8),
            srow_x=(2.0, 0.3, 0.0, 10.0),
            srow_y=(0.0, 3.0, -0.5, 20.0),
            srow_z=(0.1, 0.0, 4.0, 30.0),
        )
        for center, radius in (((15.0, 26.0, 42.0), 5.0),
                               ((12.1, 20.9, 30.4), 2.5),
                               ((10.0, 20.0, 30.0), 0.7),
                               ((0.0, 0.0, 0.0), 100.0)):
            with self.subTest(center=center, radius=radius):
                self.assert_matches_reference(vol, center, radius)

    def test_ball_extends_beyond_image_only_actual_voxels_count(self):
        vol = make_vol(dims=(4, 4, 4))
        # huge ball centred on a corner voxel: the whole image is inside
        stats = sphere_stats(vol, (0.0, 0.0, 0.0), 1000.0)
        self.assertEqual(stats["count"], 64)
        self.assertEqual(stats["transform"], "sform")

    def test_empty_region(self):
        vol = make_vol(dims=(3, 3, 3))
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (50.0, 50.0, 50.0), 0.5)
        self.assertEqual(ctx.exception.code, "empty_region")

    def test_radius_zero_misses_voxel_centers(self):
        vol = make_vol()
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (0.5, 0.5, 0.5), 0.0)
        self.assertEqual(ctx.exception.code, "empty_region")

    def test_non_finite_intensity_flagged(self):
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (2, 2, 2) else linear_fn(i, j, k)

        vol = make_vol(data_fn=nan_fn)
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (2.0, 2.0, 2.0), 1.5)
        self.assertEqual(ctx.exception.code, "non_finite_intensity")
        self.assertIn("(2, 2, 2)", str(ctx.exception))

    def test_non_finite_outside_ball_ignored(self):
        def nan_fn(i, j, k):
            return float("nan") if (i, j, k) == (4, 4, 4) else linear_fn(i, j, k)

        vol = make_vol(data_fn=nan_fn)
        stats = sphere_stats(vol, (0.0, 0.0, 0.0), 1.0)
        self.assertEqual(stats["count"], 4)

    def test_scaling_applied(self):
        vol = make_vol(datatype="int16", data_fn=lambda i, j, k: 5,
                       slope=2.0, inter=3.0)
        stats = sphere_stats(vol, (0.0, 0.0, 0.0), 1.0)
        self.assertEqual(stats["min"], 13.0)
        self.assertEqual(stats["max"], 13.0)
        self.assertEqual(stats["mean"], 13.0)

    def test_qform_source(self):
        vol = make_vol(transform="qform", quatern=(0, 0, 0),
                       qoffset=(10, 20, 30), pixdim=(1, 2, 3, 4))
        # voxel (1,1,1) under diag(2,3,4)+(10,20,30) -> (12,23,34)
        stats = sphere_stats(vol, (12.0, 23.0, 34.0), 0.0)
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["transform"], "qform")

    def test_exactly_at_capacity_succeeds(self):
        dims = (64, 64, 16)  # 65536 voxels total
        self.assertEqual(math.prod(dims), MAX_SPHERE_VOXELS)
        vol = make_vol(dims=dims, data_fn=lambda i, j, k: 1)
        stats = sphere_stats(vol, (31.5, 31.5, 7.5), 1000.0)
        self.assertEqual(stats["count"], MAX_SPHERE_VOXELS)
        self.assertEqual(stats["mean"], 1.0)

    def test_over_capacity_flagged(self):
        dims = (64, 64, 64)  # 262144 voxels
        vol = make_vol(dims=dims, data_fn=lambda i, j, k: 0)
        with self.assertRaises(RegionError) as ctx:
            sphere_stats(vol, (32.0, 32.0, 32.0), 1000.0)
        self.assertEqual(ctx.exception.code, "region_capacity_exceeded")

    def test_capacity_counts_only_actual_members(self):
        # A ball notionally larger than the cap succeeds when the image
        # itself contains at most the cap number of voxels.
        dims = (40, 40, 40)  # 64000 voxels
        vol = make_vol(dims=dims, data_fn=lambda i, j, k: 0)
        stats = sphere_stats(vol, (19.5, 19.5, 19.5), 1000.0)
        self.assertEqual(stats["count"], math.prod(dims))
        self.assertLess(stats["count"], MAX_SPHERE_VOXELS)

    def test_extreme_inputs_do_not_raise_500(self):
        vol = make_vol(dims=(3, 3, 3))
        # gigantic but finite coordinates/radius overflow intermediate
        # floats: the request must still resolve to a per-region result
        for center, radius in (((1e200, 0.0, 0.0), 1.0),
                               ((0.0, 0.0, 0.0), 1e200),
                               ((1e150, 1e150, 1e150), 1e150)):
            with self.subTest(center=center, radius=radius):
                try:
                    result = sphere_stats(vol, center, radius)
                except RegionError as err:
                    self.assertEqual(err.code, "empty_region")
                else:
                    self.assertLessEqual(result["count"], 27)


if __name__ == "__main__":
    unittest.main()
