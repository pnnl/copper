from unittest import TestCase

import copper as cp
import pickle as pkl
import numpy as np
import CoolProp.CoolProp as CP
import os

location = os.path.dirname(os.path.realpath(__file__))
DX_lib = os.path.join(location, "../copper/data", "unitarydirectexpansion_curves.json")

class UnitaryDirectExpansion(TestCase):
    lib = cp.Library(path=DX_lib)

    def test_calc_eff_ect(self):
        DX = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=DX_lib.get_set_of_curves_by_name("0").curves,
        )
        cop_1 = round(5.981486044926383,2)
        cop_2 = round(DX.calc_rated_eff(), 2)
        self.assertTrue(cop_1 == cop_2, f"{cop_1} is different than {cop_2}")