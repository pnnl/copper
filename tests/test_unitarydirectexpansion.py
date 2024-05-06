from unittest import TestCase

import copper as cp
import pickle as pkl
import numpy as np
import CoolProp.CoolProp as CP
import os

location = os.path.dirname(os.path.realpath(__file__))
DX_lib = os.path.join(location, "../copper/data", "unitarydirectexpansion_curves.json")


class UnitaryDirectExpansion(TestCase):
    def test_calc_eff_ect(self):
        lib = cp.Library(path=DX_lib)
        DX = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_gross_cap=471000,
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("0").curves,
            fan_control_mode="constant_speed",
        )
        cop_1 = 7.16
        cop_2 = round(DX.calc_rated_eff(), 2)
        self.assertTrue(cop_1 == cop_2, f"{cop_1} is different than {cop_2}")

        def test_calc_eff_ect_error(self):
            ##############
            # Todo test case for error outputs
            pass
