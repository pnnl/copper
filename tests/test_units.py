from unittest import TestCase

import copper as cp


class TestRating(TestCase):
    def test_conversion(self):
        # Efficiency conversion
        kWpton_ = cp.Units(14.0, "eer")
        self.assertTrue(round(kWpton_.conversion("kW/ton"), 3) == 0.857)
        kWpton_ = cp.Units(3.5, "cop")
        self.assertTrue(round(kWpton_.conversion("kW/ton"), 3) == 1.005)
        kWpton_ = cp.Units(0.213, "eir")
        self.assertTrue(round(kWpton_.conversion("kW/ton"), 2) == 0.75)

        eer_ = cp.Units(0.75, "kW/ton")
        self.assertTrue(round(eer_.conversion("eer"), 3) == 16.0)
        eer_ = cp.Units(3.5, "cop")
        self.assertTrue(round(eer_.conversion("eer"), 3) == 11.942)
        eer_ = cp.Units(0.213, "eir")
        self.assertTrue(round(eer_.conversion("eer"), 1) == 16.0)

        cop_ = cp.Units(0.75, "kW/ton")
        self.assertTrue(round(cop_.conversion("cop"), 3) == 4.689)
        cop_ = cp.Units(14.0, "eer")
        self.assertTrue(round(cop_.conversion("cop"), 3) == 4.103)
        cop_ = cp.Units(0.213, "eir")
        self.assertTrue(round(cop_.conversion("cop"), 1) == 4.7)

        eir_ = cp.Units(0.75, "kW/ton")
        self.assertTrue(round(eir_.conversion("eir"), 3) == 0.213)
        eir_ = cp.Units(16.0, "eer")
        self.assertTrue(round(eir_.conversion("eir"), 3) == 0.213)
        eir_ = cp.Units(4.7, "cop")
        self.assertTrue(round(eir_.conversion("eir"), 3) == 0.213)

        # Power conversion
        ton_ = cp.Units(300.0, "ton")
        self.assertTrue(round(ton_.conversion("kW"), 1) == 1055.1)

        kW_ = cp.Units(130, "kW")
        self.assertTrue(round(kW_.conversion("ton"), 3) == 36.965)

        kW_ = cp.Units(130, "kW")
        self.assertTrue(round(kW_.conversion("W"), 3) == 130000)

        W_ = cp.Units(130, "W")
        self.assertTrue(round(W_.conversion("ton"), 3) == 0.037)

        W_ = cp.Units(130, "W")
        self.assertTrue(round(W_.conversion("kW"), 3) == 0.130)

        # Temperature conversion
        degC_ = cp.Units(0, "degC")
        self.assertTrue(round(degC_.conversion("degF"), 0) == 32)

        degF_ = cp.Units(0, "degF")
        self.assertTrue(round(degF_.conversion("degC"), 2) == -17.78)
