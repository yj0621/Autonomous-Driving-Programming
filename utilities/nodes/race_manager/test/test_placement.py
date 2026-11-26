import math
import random
import unittest

from race_manager.placement import (Candidate, _matches, draw, generate_candidates,
                                    parse_recipe)


def _cand(side, severity):
    return Candidate(id='x', s=1.0, d=0.0, theta=0.0, x=0.0, y=0.0,
                     side=side, severity=severity, visibility=1.0, half_w=0.15)


class ForcedSlotTest(unittest.TestCase):
    def setUp(self):
        self.slot = parse_recipe('forced:1:0.20')[1][0]

    def test_forced_slot_parses_kind_count_and_threshold(self):
        self.assertEqual((self.slot.kind, self.slot.count, self.slot.min_severity),
                         ('forced', 1, 0.20))

    def test_forced_slot_takes_either_side_above_threshold(self):
        self.assertTrue(_matches(_cand('LEFT', 0.25), self.slot))
        self.assertTrue(_matches(_cand('RIGHT', 0.25), self.slot))

    def test_forced_slot_rejects_two_way_and_below_threshold(self):
        self.assertFalse(_matches(_cand('either', 0.25), self.slot))
        self.assertFalse(_matches(_cand('LEFT', 0.15), self.slot))

    def test_one_recipe_line_parses_every_slot(self):
        kinds = [s.kind for s in parse_recipe('chicane:1:0.10 forced:1:0.20 either:1')[1]]
        self.assertEqual(kinds, ['chicane', 'forced', 'either'])

    def test_a_slot_with_no_kind_is_rejected_instead_of_skipped(self):
        with self.assertRaises(ValueError):
            parse_recipe('chicane:1:0.10 :0.20 forced:1:0.20')


class _StraightTrack:
    """20 m of 1.4 m-wide straight, at a waypoint pitch that does not divide
    candidate_spacing — the real maps are 0.0999 m apart, and that is what makes
    the station walk drift."""

    def __init__(self):
        import numpy as np
        self.s = np.arange(0.0, 20.0, 0.0999)
        self.x = self.s.copy()
        self.y = np.zeros_like(self.s)
        self.psi = np.zeros_like(self.s)
        self.d_left = np.full_like(self.s, 0.7)
        self.d_right = np.full_like(self.s, 0.7)
        self.length = float(self.s[-1])

    def lateral(self, idx, d):
        return (self.x[idx], self.y[idx] + d)

    def arc_delta(self, s_from, s_to):
        half = self.length / 2.0
        return (s_to - s_from + half) % self.length - half

    def boundaries(self):
        import numpy as np
        return (np.stack([self.x, self.y + 0.7], axis=1),
                np.stack([self.x, self.y - 0.7], axis=1))


class ExclusionBandTest(unittest.TestCase):
    """The band is asymmetric on purpose: the re-draw fires as the car crosses
    s = 0, so only the band ahead of it can pop an obstacle into its face."""

    def _pool(self, ahead, behind):
        return generate_candidates(
            _StraightTrack(), size=0.30, wall_margin=0.15, car_width=0.31,
            squeeze_clearance=0.05, spacing=0.5, d_step=0.10, thetas=[0.0],
            exclusion_ahead=ahead, exclusion_behind=behind, min_severity=0.05)

    def test_nothing_is_placed_in_the_band_ahead_of_the_start_line(self):
        self.assertFalse([c for c in self._pool(5.0, 1.0) if c.s < 5.0])

    def test_the_band_before_the_start_line_opens_up_independently(self):
        near_finish = [c for c in self._pool(5.0, 1.0) if c.s > 15.0]
        self.assertTrue(near_finish)
        self.assertFalse([c for c in self._pool(5.0, 5.0) if c.s > 15.0])

    def test_stations_hold_the_configured_spacing_on_average(self):
        """Stepping on from the chosen s rounds up to the next waypoint every
        time, so the stations creep apart (0.5 becomes 0.6 at a 0.0999 pitch).
        Pinning them to an ideal grid keeps the mean at the configured value."""
        stations = sorted({round(c.s, 4) for c in self._pool(5.0, 1.0)})
        mean_gap = (stations[-1] - stations[0]) / (len(stations) - 1)
        self.assertAlmostEqual(mean_gap, 0.5, delta=0.02)


if __name__ == '__main__':
    unittest.main()


class SlotBudgetTest(unittest.TestCase):
    """A recipe may ask for more obstacles than num_obstacles. Truncating the
    surplus off the end splits a chicane pair and publishes a lone obstacle, so
    a slot that does not fit has to be reported the way an unfillable one is."""

    def _draw(self, spec, count, seed=0):
        track = _StraightTrack()
        cands = generate_candidates(
            track, size=0.30, wall_margin=0.15, car_width=0.31, squeeze_clearance=0.05,
            spacing=0.5, d_step=0.10, thetas=[0.0, math.pi / 4], exclusion_ahead=5.0,
            exclusion_behind=1.0, min_severity=0.0)
        spread, slots = parse_recipe(spec)
        return draw(cands, count, random.Random(seed), track, wall_margin=0.15, size=0.30,
                    min_obstacle_gap=0.50, slots=slots, chicane_span=2.5,
                    car_width=0.31, squeeze_clearance=0.05, spread=spread)

    def test_a_slot_that_does_not_fit_the_budget_is_reported(self):
        chosen, relaxed = self._draw('forced:1:0.20 chicane:1:0.10', 2)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(relaxed, 'the dropped chicane slot was not reported')

    def test_a_recipe_inside_the_budget_is_untouched(self):
        chosen, relaxed = self._draw('forced:1:0.20 chicane:1:0.10', 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(relaxed, [])


class SpreadTest(unittest.TestCase):
    """`spread:<fraction>` turns the draw into dart throwing: a minimum arc gap of
    `spread × usable_span / (groups + 1)` between separate slots. A chicane pair is
    placed in one call, so the two halves stay together whatever the spread is."""

    _draw = SlotBudgetTest._draw

    def test_spread_is_read_off_the_recipe_line(self):
        self.assertEqual(parse_recipe('spread:1.0 forced:2:0.20')[0], 1.0)
        self.assertEqual([s.kind for s in parse_recipe('spread:1.0 forced:2:0.20')[1]],
                         ['forced'])

    def test_a_recipe_without_spread_defaults_to_none(self):
        self.assertEqual(parse_recipe('forced:2:0.20')[0], 0.0)

    def test_spread_holds_a_minimum_gap_between_slots(self):
        for seed in range(5):
            chosen, _relaxed = self._draw('spread:1.0 forced:4', 4, seed)
            gaps = [b - a for a, b in zip(sorted(c.s for c in chosen),
                                          sorted(c.s for c in chosen)[1:])]
            self.assertTrue(all(g >= 2.0 for g in gaps), f'seed {seed}: {gaps}')

    def test_a_chicane_pair_stays_together_under_spread(self):
        for seed in range(5):
            chosen, _relaxed = self._draw('spread:1.0 chicane:1:0.10', 2, seed)
            self.assertEqual(len(chosen), 2)
            self.assertLessEqual(abs(chosen[0].s - chosen[1].s), 2.5)
