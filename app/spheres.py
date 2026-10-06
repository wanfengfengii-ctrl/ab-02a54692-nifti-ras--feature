"""Closed-ball intensity statistics in RAS world space.

For each region (a world-space centre and positive radius) every *voxel
centre* whose RAS coordinate (produced by the volume's selected sform or
qform affine) lies in the closed ball ``|x - c| <= r`` is admitted.  The
ball may extend beyond the image; only voxels actually present in the
volume are counted.  Intensities are the same scaling-applied values the
sampling endpoint returns (``raw * scl_slope + scl_inter``, or raw when
``scl_slope == 0``).
"""
from __future__ import annotations

import math

from .errors import RegionError

#: Maximum number of in-image voxel centres a single region may admit.
MAX_SPHERE_VOXELS = 65536

# Relative slack absorbing float64 round-off so that a voxel centre that
# lies exactly on the sphere (|x-c|^2 == r^2) is never rejected.
SURFACE_EPS = 1e-12


def sphere_stats(volume, center, radius):
    """Compute ``count/min/max/mean`` over one closed-ball region.

    Returns a dict with ``count``, ``min``, ``max``, ``mean`` and
    ``transform``.  Raises :class:`RegionError` with stable codes:

    * ``empty_region`` — no actual voxel centre lies in the ball;
    * ``region_capacity_exceeded`` — the ball admits more than
      :data:`MAX_SPHERE_VOXELS` in-image voxel centres;
    * ``non_finite_intensity`` — an admitted voxel's scaled intensity is
      not finite.
    """
    nx, ny, nz = volume.dims
    inv = volume.inverse
    x, y, z = center
    # Continuous voxel coordinate of the sphere centre.
    fi = inv[0][0] * x + inv[0][1] * y + inv[0][2] * z + inv[0][3]
    fj = inv[1][0] * x + inv[1][1] * y + inv[1][2] * z + inv[1][3]
    fk = inv[2][0] * x + inv[2][1] * y + inv[2][2] * z + inv[2][3]

    # The world ball maps to the ellipsoid {u: |L u| <= r}; the maximal
    # displacement along voxel axis a is r * |row a of L^-1|, giving a
    # conservative integer candidate box.  The +/-1 voxel margin absorbs
    # round-off; membership is always re-tested with the exact distance,
    # so the margin can only add candidates, never members.
    spans = (
        radius * math.sqrt(inv[0][0] ** 2 + inv[0][1] ** 2 + inv[0][2] ** 2),
        radius * math.sqrt(inv[1][0] ** 2 + inv[1][1] ** 2 + inv[1][2] ** 2),
        radius * math.sqrt(inv[2][0] ** 2 + inv[2][1] ** 2 + inv[2][2] ** 2),
    )
    ranges = []
    for fc, span, n in ((fi, spans[0], nx), (fj, spans[1], ny),
                        (fk, spans[2], nz)):
        edge_lo = math.ceil(fc - span)
        edge_hi = math.floor(fc + span)
        if not (math.isfinite(fc) and math.isfinite(span)
                and math.isfinite(edge_lo) and math.isfinite(edge_hi)):
            # Overflow (e.g. gigantic radius/coordinate): scan the whole
            # axis; the exact membership test still decides inclusion.
            lo, hi = 0, n - 1
        else:
            lo = max(0, int(edge_lo) - 1)
            hi = min(n - 1, int(edge_hi) + 1)
        if lo > hi:
            raise RegionError(
                "empty_region",
                "sphere contains no in-image voxel centre",
            )
        ranges.append((lo, hi))

    r2 = radius * radius
    tol = SURFACE_EPS * r2
    (a00, a01, a02, a03), (a10, a11, a12, a13), (a20, a21, a22, a23) = \
        volume.affine[0], volume.affine[1], volume.affine[2]

    count = 0
    vmin = math.inf
    vmax = -math.inf
    total = 0.0
    sum_residual = 0.0  # Kahan summation residual
    (i_lo, i_hi), (j_lo, j_hi), (k_lo, k_hi) = ranges
    for k in range(k_lo, k_hi + 1):
        kx = a02 * k + a03
        ky = a12 * k + a13
        kz = a22 * k + a23
        for j in range(j_lo, j_hi + 1):
            jx = kx + a01 * j
            jy = ky + a11 * j
            jz = kz + a21 * j
            for i in range(i_lo, i_hi + 1):
                wx = jx + a00 * i
                wy = jy + a10 * i
                wz = jz + a20 * i
                dx = wx - x
                dy = wy - y
                dz = wz - z
                d2 = dx * dx + dy * dy + dz * dz
                # Non-finite distance (overflow with extreme inputs) can
                # never be shown inside the ball; skip rather than match
                # an infinite r^2.
                if not math.isfinite(d2) or d2 > r2 + tol:
                    continue  # outside the closed ball
                if count >= MAX_SPHERE_VOXELS:
                    raise RegionError(
                        "region_capacity_exceeded",
                        f"sphere admits more than {MAX_SPHERE_VOXELS} "
                        f"in-image voxel centres",
                    )
                value = volume.value_at(i, j, k)
                if not math.isfinite(value):
                    raise RegionError(
                        "non_finite_intensity",
                        f"non-finite scaled intensity at voxel ({i}, {j}, {k})",
                    )
                count += 1
                term = value - sum_residual
                new_total = total + term
                sum_residual = (new_total - total) - term
                total = new_total
                if value < vmin:
                    vmin = value
                if value > vmax:
                    vmax = value

    if count == 0:
        raise RegionError(
            "empty_region",
            "sphere contains no in-image voxel centre",
        )

    return {
        "count": count,
        "min": vmin,
        "max": vmax,
        "mean": total / count,
        "transform": volume.transform,
    }
