from unittest import TestCase

import copper as cp


class TestRating(TestCase):
    def test_conversion(self):
        # Efficiency conversion
        kWpton_ = cp.Units(14.0, "eer")
        self.assertTrue(round(kWpton_.conversion("kw/ton"), 3) == 0.857)
        kWpton_ = cp.Units(3.5, "cop")
        self.assertTrue(round(kWpton_.conversion("kw/ton"), 3) == 1.005)

        eer_ = cp.Units(0.75, "kw/ton")
        self.assertTrue(round(eer_.conversion("eer"), 3) == 16.0)
        eer_ = cp.Units(3.5, "cop")
        self.assertTrue(round(eer_.conversion("eer"), 3) == 11.942)

        COP_ = cp.Units(0.75, "kw/ton")
        self.assertTrue(round(COP_.conversion("cop"), 3) == 4.689)
        COP_ = cp.Units(14.0, "eer")
        self.assertTrue(round(COP_.conversion("cop"), 3) == 4.103)

        # Power conversion
        ton_ = cp.Units(300.0, "ton")
        self.assertTrue(round(ton_.conversion("kW"), 3) == 1055.1)

        kW_ = cp.Units(130, "kW")
        self.assertTrue(round(kW_.conversion("ton"), 3) == 36.963)
