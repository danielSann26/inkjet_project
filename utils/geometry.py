"""Geometry helpers used by the preprocessor, canvas widget, and patch
extractor. Per Section 8 of the spec.

All functions are pure (no I/O, no globals). Points are ``(x, y)`` tuples in
image-pixel coordinates; the y-axis grows downward as in OpenCV / Qt.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


# Type aliases for readability. We don't enforce them at runtime — they're
# documentation for callers.
Point = tuple[float, float]


def angle_between_points(p1: Point, p2: Point) -> float:
    """Return the angle in degrees of the line from ``p1`` to ``p2``.

    Used by the rotate-line tool: the user clicks two points along the slide
    edge and we feed those two points here. ``compute_rotation_angle`` in
    ``preprocessor.py`` then negates this value to get the angle that brings
    the line to horizontal.

    The result is in the range ``(-180, 180]``.
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def rotate_point(point: Point, center: Point, angle_deg: float) -> Point:
    """Rotate ``point`` around ``center`` by ``angle_deg`` degrees.

    Uses the standard 2D rotation matrix. Positive ``angle_deg`` rotates
    counter-clockwise in math convention; in image-coordinate space (y-down)
    that visually appears clockwise, which matches OpenCV's convention used
    elsewhere in the pipeline.
    """
    px, py = point
    cx, cy = center
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = px - cx
    dy = py - cy
    new_x = cx + (dx * cos_a - dy * sin_a)
    new_y = cy + (dx * sin_a + dy * cos_a)
    return (new_x, new_y)


def polygon_bounding_rect(points: Sequence[Point]) -> tuple[int, int, int, int]:
    """Return the axis-aligned bounding box of ``points`` as ``(x, y, w, h)``.

    Coordinates are floored/ceiled to integer pixel bounds, which is what
    ``crop_polygon`` in the preprocessor needs to slice a NumPy array.

    Raises ``ValueError`` for an empty point list because there is no sensible
    default bounding box.
    """
    if not points:
        raise ValueError("polygon_bounding_rect requires at least one point")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = int(math.floor(min(xs)))
    y_min = int(math.floor(min(ys)))
    x_max = int(math.ceil(max(xs)))
    y_max = int(math.ceil(max(ys)))
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def non_maximum_suppression(
    candidates: Iterable[tuple],
    min_distance: float,
) -> list[tuple]:
    """Greedy NMS for fiber-intersection candidates.

    ``candidates`` is an iterable of tuples whose first three elements are
    ``(x, y, score, ...)``. Any extra trailing elements (e.g.
    ``outer_density``) are preserved untouched in returned tuples.

    Sort by score descending, then walk once, accepting each candidate that
    is at least ``min_distance`` away from every already-accepted one.

    A naive double-loop is O(n*k) — fine for hundreds of candidates but
    catastrophic at the tens-of-thousands we get on 2500x2500 fluorescence
    images. We bucket accepted candidates into a grid of cell-size
    ``min_distance``: a new candidate only needs to compare against survivors
    in its own cell and the eight neighbors. That makes the total cost
    effectively O(n) for typical density.
    """
    if min_distance <= 0:
        return list(candidates)

    items = list(candidates)
    if not items:
        return []

    # Stable sort by score (3rd element) descending.
    items.sort(key=lambda c: c[2], reverse=True)

    cell = float(min_distance)  # one bucket per `min_distance` units on a side
    min_d_sq = min_distance * min_distance

    # Grid keyed by integer cell coords -> list of (x, y) of accepted points.
    # Storing only (x, y) rather than the full tuple keeps the inner loop tight.
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    kept: list[tuple] = []

    for cand in items:
        cx, cy = cand[0], cand[1]
        gx = int(cx // cell)
        gy = int(cy // cell)

        too_close = False
        # Check the 3x3 neighborhood of cells around (gx, gy)
        for dx in (-1, 0, 1):
            if too_close:
                break
            for dy in (-1, 0, 1):
                bucket = grid.get((gx + dx, gy + dy))
                if not bucket:
                    continue
                for kx, ky in bucket:
                    ddx = cx - kx
                    ddy = cy - ky
                    if ddx * ddx + ddy * ddy < min_d_sq:
                        too_close = True
                        break
                if too_close:
                    break

        if not too_close:
            kept.append(cand)
            grid.setdefault((gx, gy), []).append((cx, cy))

    return kept
