"""Obstacle candidate generation and the seeded draw.

Kept free of rclpy so the placement rules can be tested on their own.

The pool is a pure function of the track and the generation parameters, so it is
never written to disk: the same waypoints and the same settings give the same
list in the same order, and every id decodes back to the grid cell it came from.
Freeze the inputs (waypoints md5 + parameters + seed), not the output.
"""

import math
from typing import NamedTuple

import numpy as np

LOOKBACK = 15.0
CHICANE_MIN_D = 0.10


class Candidate(NamedTuple):
    id: str
    s: float
    d: float
    theta: float        # map frame, already includes the track heading
    x: float
    y: float
    side: str           # 'either' | 'LEFT' | 'RIGHT' — the side the car may use
    severity: float     # metres the car must leave the raceline to get past
    visibility: float   # metres of raceline it stays in line of sight for
    half_w: float       # half the footprint, lateral and longitudinal alike


class Slot(NamedTuple):
    kind: str           # 'chicane' | 'LEFT' | 'RIGHT' | 'either' | 'any'
    count: int
    min_severity: float
    max_visibility: float


def half_extent(size, theta):
    """Half the footprint of a square of side `size` turned by `theta`."""
    return size / 2.0 * (abs(math.cos(theta)) + abs(math.sin(theta)))


def generate_candidates(track, size, wall_margin, car_width, squeeze_clearance,
                        spacing, d_step, thetas, exclusion_ahead, exclusion_behind,
                        min_severity):
    """Every grid placement the car actually has to react to, in a fixed order.

    Enumerated s ascending, then d ascending, then theta in the given order, so a
    candidate's position in the list depends on nothing but the waypoints and
    these arguments. `squeeze_clearance` is the air left on each side of the car
    in the tightest gap it is still expected to take.
    """
    squeeze = car_width + 2.0 * squeeze_clearance
    car_half = car_width / 2.0
    narrowest = min(half_extent(size, t) for t in thetas)
    vis = _Visibility(track)

    out = []
    next_s = exclusion_ahead
    s_idx = -1
    for i, s in enumerate(track.s):
        # Skip the start line neighbourhood and keep stations `spacing` apart
        if s < exclusion_ahead or s > track.length - exclusion_behind or s < next_s:
            continue
        s_idx += 1
        next_s = exclusion_ahead + (s_idx + 1) * spacing

        left = track.d_left[i] - wall_margin
        right = track.d_right[i] - wall_margin
        for d_idx in range(int(math.floor((narrowest - right) / d_step)),
                           int(math.ceil((left - narrowest) / d_step)) + 1):
            d = d_idx * d_step
            for t_idx, theta in enumerate(thetas):
                half = half_extent(size, theta)
                if d - half < -right or d + half > left:
                    continue

                gap_left = left - (d + half)
                gap_right = (d - half) + right
                pass_left = gap_left >= squeeze
                pass_right = gap_right >= squeeze
                if not (pass_left or pass_right):
                    continue

                # How far off the raceline the car must go, on whichever open side is cheaper
                cost_left = max(0.0, d + half + car_half) if pass_left else math.inf
                cost_right = max(0.0, half + car_half - d) if pass_right else math.inf
                severity = min(cost_left, cost_right)
                if severity < min_severity:
                    continue

                x, y = track.lateral(i, d)
                out.append(Candidate(
                    id=f's{s_idx:03d}d{d_idx:+03d}t{t_idx}',
                    s=float(s),
                    d=round(d, 4),
                    theta=float(track.psi[i]) + theta,
                    x=float(x),
                    y=float(y),
                    side='either' if pass_left and pass_right else ('LEFT' if pass_left else 'RIGHT'),
                    severity=round(severity, 3),
                    visibility=vis.of(float(x), float(y), i),
                    half_w=half,
                ))
    return out


def parse_recipe(spec):
    """One recipe: whitespace separated `kind:count[:min_severity[:max_visibility]]` slots.

    'chicane' takes two obstacles: a pair close enough in s that the car has to
    weave, offset to opposite sides of the raceline. The pseudo-slot `spread:<f>`
    is not a slot but this recipe's dart-throwing fraction; it returns alongside.
    """
    spread, slots = 0.0, []
    for tok in spec.split():
        parts = [p.strip() for p in str(tok).split(':')]
        if not parts[0]:
            raise ValueError(f'recipe slot has no kind: {tok!r}')
        if parts[0] == 'spread':
            spread = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
            continue
        slots.append(Slot(
            kind=parts[0],
            count=int(parts[1]) if len(parts) > 1 and parts[1] else 1,
            min_severity=float(parts[2]) if len(parts) > 2 and parts[2] else 0.0,
            max_visibility=float(parts[3]) if len(parts) > 3 and parts[3] else math.inf,
        ))
    return spread, slots


def draw(candidates, count, rng, track, wall_margin, size, min_obstacle_gap,
         slots, chicane_span, car_width, squeeze_clearance, spread=0.0):
    """Fill the recipe from a shuffled pool, then top up with anything that fits.

    Greedy over one shuffle, so the result is a pure function of the RNG state:
    same seed, same sequence of layouts. Returns the chosen candidates and the
    list of relaxations that were needed to fill the recipe.
    """
    order = list(candidates)
    rng.shuffle(order)
    floor = min_obstacle_gap + size
    squeeze = car_width + 2.0 * squeeze_clearance
    s_gap = spread * _band(candidates) / (_groups(slots, count) + 1)

    chosen = []
    relaxed = []

    def fits(*cands):
        trial = chosen + list(cands)
        for a in range(len(trial)):
            for b in range(a + 1, len(trial)):
                if math.dist((trial[a].x, trial[a].y), (trial[b].x, trial[b].y)) < floor:
                    return False
        for new in cands:
            for old in chosen:
                if abs(track.arc_delta(old.s, new.s)) < s_gap:
                    return False
        return _corridor(trial, track, wall_margin) >= squeeze

    for slot in slots:
        for _ in range(slot.count):
            need = 2 if slot.kind == 'chicane' else 1
            if count - len(chosen) < need:
                relaxed.append(f'{slot.kind}: dropped, no room left in num_obstacles')
                break
            taken = _fill(order, chosen, slot, fits, floor, chicane_span)
            if taken:
                chosen.extend(taken)
                continue
            # Relax the slot step by step when it cannot be filled, and record what gave
            for fallback, note in ((slot._replace(min_severity=0.0, max_visibility=math.inf),
                                    f'{slot.kind}: dropped severity/visibility limits'),
                                   (Slot('any', 1, 0.0, math.inf),
                                    f'{slot.kind}: fell back to any')):
                taken = _fill(order, chosen, fallback, fits, floor, chicane_span)
                if taken:
                    chosen.extend(taken)
                    relaxed.append(note)
                    break
            else:
                relaxed.append(f'{slot.kind}: unfilled')

    for cand in order:
        if len(chosen) >= count:
            break
        if cand not in chosen and fits(cand):
            chosen.append(cand)

    return chosen[:count], relaxed


def _band(candidates):
    """Arc length the pool actually covers, so a spread is map independent."""
    return max(c.s for c in candidates) - min(c.s for c in candidates)


def _groups(slots, count):
    """Placements that move independently: a chicane pair is one, each free obstacle is one."""
    used = sum((2 if s.kind == 'chicane' else 1) * s.count for s in slots)
    return sum(s.count for s in slots) + max(0, count - used)


def _fill(order, chosen, slot, fits, floor, chicane_span):
    if slot.kind == 'chicane':
        return _chicane(order, chosen, slot, fits, floor, chicane_span)
    for cand in order:
        if cand in chosen or not _matches(cand, slot):
            continue
        if fits(cand):
            return [cand]
    return []


def _chicane(order, chosen, slot, fits, floor, span):
    """A pair the car has to swing through in opposite directions."""
    for a in order:
        if a in chosen or not _matches(a, slot) or not _weaves(a):
            continue
        for b in order:
            if b is a or b in chosen or not _matches(b, slot) or not _weaves(b):
                continue
            if _push(a) * _push(b) > 0:
                continue
            if not (floor <= abs(a.s - b.s) <= span):
                continue
            if fits(a, b):
                return [a, b]
    return []


def _push(cand):
    """Which way the car has to swing to clear this one: +1 left, -1 right.

    Opposite d is not enough on its own -- two obstacles offset either side of
    the line can still both be walled in so that the car passes both on the
    same side, which is no chicane at all.
    """
    if cand.side == 'LEFT':
        return 1
    if cand.side == 'RIGHT':
        return -1
    return 1 if cand.d < 0 else -1


def _weaves(cand):
    """An 'either' obstacle only pushes the car aside once it is off the line."""
    return cand.side != 'either' or abs(cand.d) >= CHICANE_MIN_D


def _matches(cand, slot):
    if slot.kind in ('LEFT', 'RIGHT', 'either') and cand.side != slot.kind:
        return False
    if slot.kind == 'forced' and cand.side == 'either':
        return False
    return cand.severity >= slot.min_severity and cand.visibility <= slot.max_visibility


def _corridor(chosen, track, wall_margin):
    """Widest lateral gap left anywhere the chosen obstacles reach.

    draw() only compares centres, so without this two obstacles could sit side by
    side and close the track between them.
    """
    worst = math.inf
    stations = sorted({round(c.s + off, 2) for c in chosen
                       for off in (-c.half_w, 0.0, c.half_w)})
    for s in stations:
        i = int(np.argmin(np.abs(track.s - s)))
        free = [(-(track.d_right[i] - wall_margin), track.d_left[i] - wall_margin)]
        for o in chosen:
            if abs(o.s - s) > o.half_w:
                continue
            lo, hi = o.d - o.half_w, o.d + o.half_w
            cut = []
            for a, b in free:
                if hi <= a or lo >= b:
                    cut.append((a, b))
                    continue
                if lo > a:
                    cut.append((a, lo))
                if hi < b:
                    cut.append((hi, b))
            free = cut
        worst = min(worst, max((b - a for a, b in free), default=0.0))
    return worst


class _Visibility:
    """How far back along the raceline a point stays in line of sight.

    The track boundaries stand in for the walls, which is the same approximation
    the corridor rules already make. Capped at LOOKBACK — past that the beam
    range and the planner horizon matter more than the geometry.
    """

    def __init__(self, track):
        left, right = track.boundaries()
        walls = np.concatenate([np.stack([left[:-1], left[1:]], axis=1),
                                np.stack([right[:-1], right[1:]], axis=1)])
        self.wa = walls[:, 0][None, :, :]
        self.wb = walls[:, 1][None, :, :]
        self.track = track
        self.cache = {}

    def of(self, x, y, idx):
        key = (idx, round(x, 3), round(y, 3))
        if key in self.cache:
            return self.cache[key]

        s = self.track.s
        arcs = s[idx] - s
        arcs[arcs < 0] += self.track.length
        window = np.flatnonzero((arcs > 0) & (arcs <= LOOKBACK))
        if window.size == 0:
            return 0.0
        window = window[np.argsort(arcs[window])]

        views = np.stack([self.track.x[window], self.track.y[window]], axis=1)[:, None, :]
        point = np.array([[[x, y]]])
        d1 = _orient(self.wa, self.wb, views)
        d2 = _orient(self.wa, self.wb, point)
        d3 = _orient(views, point, self.wa)
        d4 = _orient(views, point, self.wb)
        hit = np.any(((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0)), axis=1)

        first = np.flatnonzero(hit)
        n = int(first[0]) if first.size else len(window)
        out = round(float(arcs[window[n - 1]]) if n else 0.0, 2)
        self.cache[key] = out
        return out


def _orient(a, b, c):
    return ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
