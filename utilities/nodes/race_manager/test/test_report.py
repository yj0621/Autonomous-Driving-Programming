import matplotlib
import unittest

matplotlib.use('Agg')

from race_manager.report import _gap_label, _sheet_layout


def _laps(count, best_lap=1):
    return [
        {
            'lap': lap,
            'time': 10.0 if lap == best_lap else 10.0 + lap / 10.0,
            'obstacle': lap % 3,
            'wall': lap % 2,
            'total': lap % 3 + lap % 2,
        }
        for lap in range(1, count + 1)
    ]


class ReportLayoutTest(unittest.TestCase):
    def test_gap_label_marks_best_and_formats_delta(self):
        self.assertEqual(_gap_label({'time': 5.869}, 5.869), 'BEST')
        self.assertEqual(_gap_label({'time': 6.241}, 5.869), '+0.37')

    def test_gap_label_signs_a_lap_quicker_than_the_best(self):
        # A lap with a collision can still be quick; it just cannot be the best
        self.assertEqual(_gap_label({'time': 5.500}, 5.869), '-0.37')

    def test_gap_label_calls_out_a_lap_timed_from_a_recovery(self):
        self.assertEqual(_gap_label({'time': 4.100, 'partial': True}, 5.869), 'PART')

    def test_gap_label_without_a_best_lap_yet(self):
        self.assertEqual(_gap_label({'time': 6.241}, None), '—')

    def test_sheet_layout_switches_to_two_columns_at_eleven_laps(self):
        ten = _sheet_layout(_laps(10), 10.0)
        eleven = _sheet_layout(_laps(11), 10.0)

        self.assertEqual(ten.columns, 1)
        self.assertEqual(eleven.columns, 2)
        self.assertEqual(eleven.laps, _laps(11))

    def test_sheet_layout_limits_long_sessions_to_latest_24_laps(self):
        layout = _sheet_layout(_laps(40, best_lap=7), 10.0)

        self.assertEqual(layout.columns, 2)
        self.assertEqual([lap['lap'] for lap in layout.laps], list(range(17, 41)))
        self.assertEqual(layout.range_label, 'SHOWING LAPS 17–40')
        self.assertEqual(layout.pinned_best['lap'], 7)

    def test_sheet_layout_does_not_pin_best_when_it_is_visible(self):
        layout = _sheet_layout(_laps(25, best_lap=25), 10.0)

        self.assertIsNone(layout.pinned_best)
        self.assertEqual(layout.range_label, 'SHOWING LAPS 2–25')


if __name__ == '__main__':
    unittest.main()
