from unittest import TestCase
from copper.units import Units

# TODO: look into adding hypothesis to the code to do range checking.


class TestRating(TestCase):
    def setUp(self) -> None:
        """Runs before every test. Good place to initialize values and store common objects."""
        self.tolerance = 3

    def tearDown(self) -> None:
        """Runs after every test and cleans up file created from the tests."""
        pass

    def test_conversion(self):
        # Efficiency conversion
        kWpton_ = Units(14.0, "eer")
        self.assertTrue(round(kWpton_.conversion("kw/ton"), 3) == 0.857)
        kWpton_ = Units(3.5, "cop")
        self.assertTrue(round(kWpton_.conversion("kw/ton"), 3) == 1.005)

        eer_ = Units(0.75, "kw/ton")
        self.assertTrue(round(eer_.conversion("eer"), 3) == 16.0)
        eer_ = Units(3.5, "cop")
        self.assertTrue(round(eer_.conversion("eer"), 3) == 11.942)

        COP_ = Units(0.75, "kw/ton")
        self.assertTrue(round(COP_.conversion("cop"), 3) == 4.689)
        COP_ = Units(14.0, "eer")
        self.assertTrue(round(COP_.conversion("cop"), 3) == 4.103)

        # Power conversion
        ton_ = Units(300.0, "ton")
        self.assertTrue(round(ton_.conversion("kW"), 3) == 1055.1)

        kW_ = Units(130, "kW")
        self.assertTrue(round(kW_.conversion("ton"), 3) == 36.963)

        kW_ = Units(130, "kW")
        self.assertTrue(round(kW_.conversion("W"), 3) == 130000)

        W_ = Units(130, "W")
        self.assertTrue(round(W_.conversion("ton"), 3) == 0.037)

        W_ = Units(130, "W")
        self.assertTrue(round(W_.conversion("kW"), 3) == 0.130)

        # Temperature conversion
        degC_ = Units(0, "degC")
        self.assertTrue(round(degC_.conversion("degF"), 0) == 32)

        degF_ = Units(0, "degF")
        self.assertTrue(round(degF_.conversion("degC"), 2) == -17.78)
