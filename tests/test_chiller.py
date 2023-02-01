from unittest import TestCase

import copper as cp
import pickle as pkl
import numpy as np
import CoolProp.CoolProp as CP
import os

location = os.path.dirname(os.path.realpath(__file__))
chiller_lib = os.path.join(location, "../copper/lib", "chiller_curves.json")


class TestChiller(TestCase):
    lib = cp.Library(path=chiller_lib)

    def test_get_reference_variable(self):
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
            set_of_curves=self.lib.get_set_of_curves_by_name("337").curves,
        )

        self.assertTrue(
            [6.7, 34.6] == [round(v, 1) for v in chlr.get_ref_values("cap-f-t")]
        )
        self.assertTrue(
            [6.7, 34.6] == [round(v, 1) for v in chlr.get_ref_values("eir-f-t")]
        )
        self.assertTrue(
            [34.6, 1.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-plr")]
        )

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
            set_of_curves=self.lib.get_set_of_curves_by_name("337").curves,
        )

        self.assertTrue(
            [7.0, 35.0] == [round(v, 1) for v in chlr.get_ref_values("cap-f-t")]
        )
        self.assertTrue(
            [7.0, 35.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-t")]
        )
        self.assertTrue(
            [35.0, 1.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-plr")]
        )

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
            set_of_curves=self.lib.get_set_of_curves_by_name("337").curves,
        )

        self.assertTrue(
            [6.7, 29.4] == [round(v, 1) for v in chlr.get_ref_values("cap-f-t")]
        )
        self.assertTrue(
            [6.7, 29.4] == [round(v, 1) for v in chlr.get_ref_values("eir-f-t")]
        )
        self.assertTrue(
            [1.0, 0.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-plr")]
        )

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
            set_of_curves=self.lib.get_set_of_curves_by_name("337").curves,
        )

        self.assertTrue(
            [7.0, 30.0] == [round(v, 1) for v in chlr.get_ref_values("cap-f-t")]
        )
        self.assertTrue(
            [7.0, 30.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-t")]
        )
        self.assertTrue(
            [1.0, 0.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-plr")]
        )

        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_550/590",
            model="ect_lwt",
            sim_engine="energyplus",
            set_of_curves=self.lib.get_set_of_curves_by_name("337").curves,
        )

        self.assertTrue(
            [6.7, 35.0] == [round(v, 1) for v in chlr.get_ref_values("cap-f-t")]
        )
        self.assertTrue(
            [6.7, 35.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-t")]
        )
        self.assertTrue(
            [1.0, 0.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-plr")]
        )

        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_551/591",
            model="ect_lwt",
            sim_engine="energyplus",
            set_of_curves=self.lib.get_set_of_curves_by_name("337").curves,
        )

        self.assertTrue(
            [7.0, 35.0] == [round(v, 1) for v in chlr.get_ref_values("cap-f-t")]
        )
        self.assertTrue(
            [7.0, 35.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-t")]
        )
        self.assertTrue(
            [1.0, 0.0] == [round(v, 1) for v in chlr.get_ref_values("eir-f-plr")]
        )

    def test_get_lct(self):
        curves = pkl.load(open("./tests/data/agg_curves.pkl", "rb"))

        chlr = cp.Chiller(
            compressor_type="screw",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=75.0,
            ref_cap_unit="ton",
            full_eff=0.79,
            full_eff_unit="kw/ton",
            part_eff=0.676,
            part_eff_unit="kw/ton",
            part_eff_ref_std="ahri_550/590",
            min_unloading=0.1,
            model="lct_lwt",
            sim_engine="energyplus",
            set_of_curves=curves,
        )

        m = chlr.get_ref_cond_flow_rate()
        cop = cp.Units(value=chlr.full_eff, unit=chlr.full_eff_unit)
        cop = cop.conversion(new_unit="cop")

        # Check that the correct condenser flow is calculated
        self.assertTrue(round(m, 3) == 0.015, f"Calculated condenser flow {m} m3/s")

        # Determine the specific heat capacity of water [kJ/kg.K]
        c_p = (
            CP.PropsSI(
                "C",
                "P",
                101325,
                "T",
                0.5 * (chlr.ref_ect + chlr.ref_lct) + 273.15,
                "Water",
            )
            / 1000
        )

        # Determine the density of water [kg/m3]
        rho = CP.PropsSI(
            "D", "P", 101325, "T", 0.5 * (chlr.ref_ect + chlr.ref_lct) + 273.15, "Water"
        )

        args = [
            chlr.ref_lwt,
            curves[1],  # cap-f-t
            curves[0],  # eir-f-t
            curves[2],  # eir-f-plr
            1,
            -999,
            cop,
            chlr.ref_ect,
            m * rho,
            c_p,
        ]

        lct = chlr.get_lct(chlr.ref_ect, args)

        # Check that the correct LCT is calculated
        self.assertTrue(
            round(lct, 2) == round(chlr.ref_lct, 2),
            f"Calculated LCT: {lct}. It must be the same as the reference LCT which is {round(chlr.ref_lct, 2)}",
        )

        # Check that full load efficiency is returned when a different unit is specified
        self.assertTrue(round(chlr.calc_rated_eff("full", unit="cop"), 2) == 5.15)

    def test_calc_eff_ect(self):
        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=6,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_550/590",
            model="ect_lwt",
            sim_engine="energyplus",
            set_of_curves=self.lib.get_set_of_curves_by_name("14").curves,
        )

        cop_1 = round(
            1
            / chlr.calc_eff_ect(
                chlr.set_of_curves[2],
                chlr.set_of_curves[0],
                chlr.set_of_curves[1],
                1 / 6,
                (85.0 - 32.0) * 5 / 9,
                (44.0 - 32.0) * 5 / 9,
                1,
            ),
            2,
        )
        cop_2 = round(chlr.calc_rated_eff("full", "cop"), 2)
        self.assertTrue(cop_1 == cop_2, f"{cop_1} is different than {cop_2}")

    def test_curves_fromm_lib(self):
        full_eff_target = 0.55
        part_eff_target = 0.38

        TestChlr = cp.Chiller(
            ref_cap=400,
            ref_cap_unit="ton",
            full_eff=full_eff_target,
            full_eff_unit="kw/ton",
            part_eff=part_eff_target,
            part_eff_unit="kw/ton",
            sim_engine="energyplus",
            model="ect_lwt",
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
        )

        lib, filters = TestChlr.get_lib_and_filters()
        csets = TestChlr.get_curves_from_lib(lib=lib, filters=filters)

        list_of_seed_bools = []

        for cs in csets:
            cond_type = cs.eqp.condenser_type
            comp_type = cs.eqp.compressor_type
            comp_speed = cs.eqp.compressor_speed

            if (
                cond_type == "water"
                and comp_type == "centrifugal"
                and comp_speed == "constant"
            ):
                list_of_seed_bools.append(True)
            else:
                list_of_seed_bools.append(False)

        seed_curve_check = np.asarray(list_of_seed_bools)
        self.assertTrue(np.all(seed_curve_check) == True)
