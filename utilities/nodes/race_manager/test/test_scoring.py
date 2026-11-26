import unittest

from race_manager.scorer import best_lap_time


def _lap(number, time, hits=0, partial=False):
    return {'lap': number, 'time': time, 'obstacle': hits, 'wall': 0,
            'total': hits, 'partial': partial}


class BestLapTest(unittest.TestCase):
    def test_picks_the_fastest_clean_lap(self):
        laps = [_lap(1, 9.71), _lap(2, 9.07), _lap(3, 9.43)]

        self.assertEqual(best_lap_time(laps), 9.07)

    def test_a_lap_with_a_collision_cannot_be_the_best(self):
        laps = [_lap(1, 7.75, hits=1), _lap(2, 9.07)]

        self.assertEqual(best_lap_time(laps), 9.07)

    def test_a_lap_timed_from_a_recovery_cannot_be_the_best(self):
        # Clean, but the clock started where the car was put back on track
        laps = [_lap(1, 7.75, partial=True), _lap(2, 9.07)]

        self.assertEqual(best_lap_time(laps), 9.07)

    def test_no_best_until_one_lap_qualifies(self):
        self.assertIsNone(best_lap_time([]))
        self.assertIsNone(best_lap_time([_lap(1, 7.75, hits=1),
                                         _lap(2, 8.10, partial=True)]))

    def test_logs_written_before_partial_was_recorded_still_read(self):
        old = [{'lap': 1, 'time': 9.71, 'obstacle': 0, 'wall': 0, 'total': 0}]

        self.assertEqual(best_lap_time(old), 9.71)


if __name__ == '__main__':
    unittest.main()
