import unittest
import numpy as np
from inno_medical_bedflow import core


class TestBedflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pts = core.simulate(2)

    def test_stays_valid(self):
        for p in self.pts[:500]:
            self.assertGreaterEqual(p["los"], 1); self.assertEqual(p["discharge"], p["admit"] + p["los"])

    def test_forecast_shape_nonnegative(self):
        f = core.forecast_census(self.pts, "surgery", 250)
        self.assertEqual(len(f), 7); self.assertTrue((f >= 0).all())

    def test_forecast_uses_only_past(self):
        # Removing all future admissions must not change the forecast.
        T = 250
        past = [p for p in self.pts if p["admit"] <= T]
        a = core.forecast_census(self.pts, "icu", T); b = core.forecast_census(past, "icu", T)
        self.assertTrue(np.allclose(a, b))

    def test_overload_alert_days(self):
        self.assertEqual(core.overload_alerts([5, 12, 13], 12), [3])

    def test_candidates_are_inpatients_and_ready(self):
        for p in core.discharge_candidates(self.pts, "icu", 300):
            self.assertTrue(p["admit"] <= 300 < p["discharge"] and p["ready_from"] <= 300)
