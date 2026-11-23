"""Track geometry read straight from a map's global_waypoints.json.

The harness deliberately does not subscribe to /global_waypoints: it has to work
even when the team's stack does not run a global planner.
"""

import json

import numpy as np


class Track:
    def __init__(self, waypoints_file: str):
        with open(waypoints_file) as f:
            data = json.load(f)
        wpnts = data['global_traj_wpnts_iqp']['wpnts']

        self.file = waypoints_file
        self.s = np.array([w['s_m'] for w in wpnts])
        self.x = np.array([w['x_m'] for w in wpnts])
        self.y = np.array([w['y_m'] for w in wpnts])
        self.psi = np.array([w['psi_rad'] for w in wpnts])
        self.d_right = np.array([w['d_right'] for w in wpnts])
        self.d_left = np.array([w['d_left'] for w in wpnts])
        self.length = float(self.s[-1])

    def nearest_s(self, x, y):
        """Arc length of the closest waypoint. Used to tell a real lap from a shortcut."""
        i = int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))
        return float(self.s[i])

    def arc_delta(self, s_from, s_to):
        """Signed shortest arc from one s to another, so the s = 0 seam is not a jump."""
        half = self.length / 2.0
        return (s_to - s_from + half) % self.length - half

    def lateral(self, idx, d):
        """Cartesian point d metres to the left of waypoint idx (+d is left)."""
        psi = self.psi[idx]
        return (self.x[idx] - d * np.sin(psi), self.y[idx] + d * np.cos(psi))

    def boundaries(self):
        """Left and right boundary polylines, for plotting."""
        left = np.array([self.lateral(i, self.d_left[i]) for i in range(len(self.s))])
        right = np.array([self.lateral(i, -self.d_right[i]) for i in range(len(self.s))])
        return left, right

    def finish_line(self):
        """Segment across the track at s = 0, plus the forward direction."""
        left = self.lateral(0, self.d_left[0])
        right = self.lateral(0, -self.d_right[0])
        heading = (float(np.cos(self.psi[0])), float(np.sin(self.psi[0])))
        return np.array(right), np.array(left), heading


def crossed_forward(p0, p1, line_a, line_b, heading):
    """True if moving p0 -> p1 crosses the line segment in the forward direction."""
    travel = (p1[0] - p0[0], p1[1] - p0[1])
    if travel[0] * heading[0] + travel[1] * heading[1] <= 0:
        return False
    return _segments_intersect(p0, p1, line_a, line_b)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, p3, p4):
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
