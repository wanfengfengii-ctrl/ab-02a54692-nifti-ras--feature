"""Closed-ball local-intensity statistics in scanner RAS world space.

For every requested region a sphere is adjudicated *in the chosen sform/qform
world space*: an integer voxel center belongs to the region iff the RAS point
the affine maps it to lies in the closed ball (surface included).  The sphere
may extend beyond the image; only real voxel centers are counted.

Two geometric failures are reported per region (never failing the whole
request):

* ``empty_region``      — no voxel center lies in the ball
* ``region_too_large``  — more than :data:`MAX_SPHERE_VOXELS` centers qualify

A non-finite *scaled* intensity inside the ball additionally fails the region
with ``non_finite_data``; min/max/mean are only defined over finite values.
"""
from __future__ import annotations

import math

from .errors import RegionError

#: Hard cap on voxel centers admitted by a single region.
MAX_SPHERE_VOXELS = 65536

# Round-off guards.  BOX_TOL (voxels) only pads the conservative candidate
# bounding box; the world-space distance test adjudicates.  The surface
# tolerance is in distance units and scales with the world magnitudes because
# dx = (A u) - c cancels two large numbers when the header carries large
# translations; ~32 ulp covers the float32-header -> float64 math.  Distance
# is evaluated with math.hypot, which never overflows for finite inputs.
BOX_TOL = 1e-6
DIST_EPS = 32.0 * 2.0 ** -53
DIST_ABS_TOL = 1e-12


def _world_to_voxel(inv, x, y, z):
    return (
        inv[0][0] * x + inv[0][1] * y + inv[0][2] * z + inv[0][3],
        inv[1][0] * x + inv[1][1] * y + inv[1][2] * z + inv[1][3],
        inv[2][0] * x + inv[2][1] * y + inv[2][2] * z + inv[2][3],
    )


def sphere_stats(volume, center, radius):
    """Compute ``count/min/max/mean`` over the ball around ``center``.

    ``center`` is an RAS world point, ``radius`` a positive world-space
    radius, both in the units of the selected affine.  Intensities use the
    file's ``scl_slope`` / ``scl_inter`` scaling rule (slope 0 -> unscaled).

    Returns ``{"count", "min", "max", "mean"}``.  Raises
    :class:`RegionError` for an empty ball, an over-capacity ball or
    non-finite scaled data among the admitted voxels.
    """
    cx, cy, cz = center
    inv = volume.inverse
    affine = volume.affine

    # Continuous voxel coordinate of the sphere centre, plus the half-width
    # (in voxels) of its image on each voxel axis.  With v = A u the voxel
    # displacement is inv(A) times the world displacement, bounded by the
    # row norm of the inverse -- a conservative candidate box.
    fi, fj, fk = _world_to_voxel(inv, cx, cy, cz)
    h_i = radius * math.sqrt(inv[0][0] ** 2 + inv[0][1] ** 2 + inv[0][2] ** 2)
    h_j = radius * math.sqrt(inv[1][0] ** 2 + inv[1][1] ** 2 + inv[1][2] ** 2)
    h_k = radius * math.sqrt(inv[2][0] ** 2 + inv[2][1] ** 2 + inv[2][2] ** 2)

    nx, ny, nz = volume.dims
    def _axis_bounds(fc, half, n):
        # Extreme (but finite) JSON inputs through a near-singular affine may
        # make the center image, half-width, or their sum/difference lose
        # finiteness: widen the candidate box to the whole axis rather than
        # raise or silently narrow it; the world-space distance test below
        # remains the exact adjudicator.
        lo = fc - half - BOX_TOL
        hi = fc + half + BOX_TOL
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return 0, n - 1
        return (max(0, math.ceil(lo)), min(n - 1, math.floor(hi)))

    i_lo, i_hi = _axis_bounds(fi, h_i, nx)
    j_lo, j_hi = _axis_bounds(fj, h_j, ny)
    k_lo, k_hi = _axis_bounds(fk, h_k, nz)

    if i_lo > i_hi or j_lo > j_hi or k_lo > k_hi:
        raise RegionError(
            "empty_region",
            f"sphere centered at ({cx!r}, {cy!r}, {cz!r}) with radius "
            f"{radius!r} does not meet the image voxel-center domain",
        )

    # O(1) capacity fast path.  With sigma_max the largest singular value of
    # the affine, ||A d|| <= sigma_max ||d||, so the voxel-space sphere of
    # radius r/sigma_max is fully inscribed in the (generally ellipsoidal)
    # image of the world ball; its inscribed axis-aligned box has half-width
    # r/(sqrt(3)*sigma_max) on every axis.  sigma_max <= Frobenius norm, a
    # cheap conservative stand-in.  The lattice count of that box is a lower
    # bound on membership, so over-capacity regions fail immediately even
    # on very fine grids instead of scanning every candidate center.
    # Skipped when the center image itself lost finiteness; the exact scan
    # still adjudicates correctly.
    if math.isfinite(fi) and math.isfinite(fj) and math.isfinite(fk):
        frob = math.sqrt(sum(v * v for row in affine[:3] for v in row[:3]))
        if frob > 0.0:
            q = radius / (math.sqrt(3.0) * frob)
        else:
            q = 0.0
        if not math.isfinite(q):
            lb_i, ub_i = 0, nx - 1
            lb_j, ub_j = 0, ny - 1
            lb_k, ub_k = 0, nz - 1
        else:
            lb_i = max(0, math.ceil(fi - q))
            ub_i = min(nx - 1, math.floor(fi + q))
            lb_j = max(0, math.ceil(fj - q))
            ub_j = min(ny - 1, math.floor(fj + q))
            lb_k = max(0, math.ceil(fk - q))
            ub_k = min(nz - 1, math.floor(fk + q))
        if lb_i <= ub_i and lb_j <= ub_j and lb_k <= ub_k:
            lower_bound = (ub_i - lb_i + 1) * (ub_j - lb_j + 1) \
                * (ub_k - lb_k + 1)
            if lower_bound > MAX_SPHERE_VOXELS:
                raise RegionError(
                    "region_too_large",
                    f"sphere centered at ({cx!r}, {cy!r}, {cz!r}) with radius "
                    f"{radius!r} encloses more than {MAX_SPHERE_VOXELS} voxel "
                    f"centers",
                )

    (a00, a01, a02, tx), (a10, a11, a12, ty), (a20, a21, a22, tz) = \
        affine[0], affine[1], affine[2]
    scale = max(abs(cx), abs(cy), abs(cz), radius)
    # Surface slack in distance units: covers the ~ulp gap when A*u and c are
    # large but nearly equal (float32 header reconstructed in float64).
    tol = DIST_EPS * (radius + scale) + DIST_ABS_TOL

    # Pass 1: adjudicate membership strictly in world space; remember linear
    # voxel indices.  Capacity is a geometric property of the region, so it
    # is settled before any intensity is read.
    indices = []
    rlimit = radius + tol
    for k in range(k_lo, k_hi + 1):
        wx_k = a02 * k + tx
        wy_k = a12 * k + ty
        wz_k = a22 * k + tz
        for j in range(j_lo, j_hi + 1):
            wx_jk = a01 * j + wx_k
            wy_jk = a11 * j + wy_k
            wz_jk = a21 * j + wz_k
            for i in range(i_lo, i_hi + 1):
                dx = a00 * i + wx_jk - cx
                dy = a10 * i + wy_jk - cy
                dz = a20 * i + wz_jk - cz
                if math.hypot(dx, dy, dz) <= rlimit:
                    if len(indices) == MAX_SPHERE_VOXELS:
                        raise RegionError(
                            "region_too_large",
                            f"sphere centered at ({cx!r}, {cy!r}, {cz!r}) with "
                            f"radius {radius!r} encloses more than "
                            f"{MAX_SPHERE_VOXELS} voxel centers",
                        )
                    indices.append(i + nx * (j + ny * k))

    if not indices:
        raise RegionError(
            "empty_region",
            f"sphere centered at ({cx!r}, {cy!r}, {cz!r}) with radius "
            f"{radius!r} contains no voxel centers",
        )

    # Pass 2: scaled intensities of the admitted centers.
    data = volume.data
    total = 0.0
    vmin = math.inf
    vmax = -math.inf
    nxy = nx * ny
    for idx in indices:
        value = volume.scale_value(data[idx])
        if not math.isfinite(value):
            k, rem = divmod(idx, nxy)
            j, i = divmod(rem, nx)
            raise RegionError(
                "non_finite_data",
                f"non-finite scaled data at voxel ({i}, {j}, {k}) inside the "
                f"sphere centered at ({cx!r}, {cy!r}, {cz!r})",
                voxel=(i, j, k),
            )
        total += value
        if value < vmin:
            vmin = value
        if value > vmax:
            vmax = value

    return {"count": len(indices), "min": vmin, "max": vmax,
            "mean": total / len(indices)}
