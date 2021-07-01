from unittest import TestCase

import copper as cp


class TestLibrary(TestCase):
    def part_load_efficiency_calcs(self):
        """
        Test part load calculations when the library is loaded.
        """

        # Load library
        lib = cp.Library(path="./fixtures/chiller_curves.json")
        self.assertTrue(
            lib.content()["ElectricEIRChiller Trane ACRA 256.8kW/3.355COP/VSD"][
                "part_eff"
            ]
            > 0
        )
