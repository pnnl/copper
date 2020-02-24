from unittest import TestCase

import copper as cp


class TestRating(TestCase):
    def test_rating_conversion(self):
        kWpton_ = cp.Rating(14.0, "EER")
        self.assertTrue(round(kWpton_.conversion("kWpton"), 3) == 0.857)
        kWpton_ = cp.Rating(3.5, "COP")
        self.assertTrue(round(kWpton_.conversion("kWpton"), 3) == 1.005)

        EER_ = cp.Rating(0.75, "kWpton")
        self.assertTrue(round(EER_.conversion("EER"), 3) == 16.0)
        EER_ = cp.Rating(3.5, "COP")
        self.assertTrue(round(EER_.conversion("EER"), 3) == 11.942)

        COP_ = cp.Rating(0.75, "kWpton")
        self.assertTrue(round(COP_.conversion("COP"), 3) == 4.689)
        COP_ = cp.Rating(14.0, "EER")
        self.assertTrue(round(COP_.conversion("COP"), 3) == 4.103)
