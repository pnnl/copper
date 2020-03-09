from unittest import TestCase

import copper as cp


class TestRating(TestCase):
    def test_conversion(self):
        # Efficiency conversion
        kWpton_ = cp.Unit(14.0, "EER")
        self.assertTrue(round(kWpton_.conversion("kWpton"), 3) == 0.857)
        kWpton_ = cp.Unit(3.5, "COP")
        self.assertTrue(round(kWpton_.conversion("kWpton"), 3) == 1.005)

        EER_ = cp.Unit(0.75, "kWpton")
        self.assertTrue(round(EER_.conversion("EER"), 3) == 16.0)
        EER_ = cp.Unit(3.5, "COP")
        self.assertTrue(round(EER_.conversion("EER"), 3) == 11.942)

        COP_ = cp.Unit(0.75, "kWpton")
        self.assertTrue(round(COP_.conversion("COP"), 3) == 4.689)
        COP_ = cp.Unit(14.0, "EER")
        self.assertTrue(round(COP_.conversion("COP"), 3) == 4.103)

        # Power conversion
        ton_ = cp.Unit(300.0, "ton")
        self.assertTrue(round(ton_.conversion("kW"), 3) == 1055.1)

        kW_ = cp.Unit(130, "kW")
        self.assertTrue(round(kW_.conversion("ton"), 3) == 36.963)
