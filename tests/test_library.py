from unittest import TestCase

import copper as cp
import os

location = os.path.dirname(os.path.realpath(__file__))
chiller_lib = os.path.join(location, "../copper/lib", "chiller_curves.json")


class TestLibrary(TestCase):
    def test_part_load_efficiency_calcs(self):
        """
        Test part load calculations when the library is loaded.
        """

        # Load library
        lib = cp.Library(path=chiller_lib)
        self.assertTrue(lib.content()["6"]["part_eff"] > 0)

        # Check calculation for the chiller EIR model
        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_551/591",
            model="ect_lwt",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("6").curves,
        )

        assert round(chlr.calc_rated_eff("part", "cop"), 2) == 5.44  # IPLV.SI

        # Check calculation for the chiller EIR model
        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_550/590",
            model="ect_lwt",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("6").curves,
        )

        assert round(chlr.calc_rated_eff("part", "cop"), 2) == 5.47  # IPLV.IP

        # Check calculation for the reformulated chiller EIR model
        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_551/591",
            model="lct_lwt",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("337").curves,
        )

        assert round(chlr.calc_rated_eff("part", "cop"), 2) == 8.22  # IPLV.SI

        # Check calculation for the reformulated chiller EIR model
        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_550/590",
            model="lct_lwt",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("337").curves,
        )

        assert round(chlr.calc_rated_eff("part", "cop"), 2) == 8.19  # IPLV.IP
