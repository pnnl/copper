from unittest import TestCase

import copper as cp
import pickle as pkl
import numpy as np
import CoolProp.CoolProp as CP
import os

location = os.path.dirname(os.path.realpath(__file__))
DX_lib = os.path.join(location, "../copper/data", "unitarydirectexpansion_curves.json")


class UnitaryDirectExpansion(TestCase):
    # Load curve library
    lib = cp.Library(path=DX_lib)

    def test_calc_eff_ect(self):
        lib = cp.Library(path=DX_lib)
        DX = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="si",
            ref_gross_cap=471000,
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
        )
        ieer = round(DX.calc_rated_eff(), 1)
        self.assertTrue(7.4 == ieer, f"{ieer} is different than 7.4")

        # Two-speed fan unit
        DX = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="si",
            ref_gross_cap=471000,
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
            indoor_fan_speeds=2,
        )
        ieer_two_spd = round(DX.calc_rated_eff(), 2)
        assert ieer_two_spd > ieer

    def test_multi_speed(self):
        # Load curve library
        lib = cp.Library(path=DX_lib)

        # Define equipment characteristics
        # Two-speed fan
        dx_unit = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="si",
            ref_gross_cap=471000,
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
            indoor_fan_speeds=2,
        )
        assert (
            dx_unit.calc_fan_power(capacity_ratio=0.5) / dx_unit.indoor_fan_power == 0.4
        )
        assert (
            dx_unit.calc_fan_power(capacity_ratio=1.0) / dx_unit.indoor_fan_power == 1.0
        )
        assert (
            dx_unit.calc_fan_power(capacity_ratio=0.75) / dx_unit.indoor_fan_power
            == 0.7
        )

        # Define equipment characteristics
        # Four-speed fan
        dx_unit = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="si",
            ref_gross_cap=471000,
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
            indoor_fan_speeds=4,
            indoor_fan_speeds_mapping={
                "1": {
                    "fan_flow_fraction": 0.2,
                    "fan_power_fraction": 0.15,
                    "capacity_fraction": 0.2,
                },
                "2": {
                    "fan_flow_fraction": 0.45,
                    "fan_power_fraction": 0.4,
                    "capacity_fraction": 0.45,
                },
                "3": {
                    "fan_flow_fraction": 0.75,
                    "fan_power_fraction": 0.7,
                    "capacity_fraction": 0.75,
                },
                "4": {
                    "fan_flow_fraction": 1.0,
                    "fan_power_fraction": 1.0,
                    "capacity_fraction": 1.0,
                },
            },
        )
        assert (
            dx_unit.calc_fan_power(capacity_ratio=0.1) / dx_unit.indoor_fan_power
            == 0.15
        )
        assert (
            dx_unit.calc_fan_power(capacity_ratio=1.0) / dx_unit.indoor_fan_power == 1.0
        )
        assert (
            dx_unit.calc_fan_power(capacity_ratio=0.75) / dx_unit.indoor_fan_power
            == 0.7
        )
        assert (
            round(
                dx_unit.calc_fan_power(capacity_ratio=0.58) / dx_unit.indoor_fan_power,
                2,
            )
            == 0.53
        )
        assert (
            round(
                dx_unit.calc_fan_power(capacity_ratio=0.70) / dx_unit.indoor_fan_power,
                2,
            )
            == 0.65
        )

    def test_generation(self):
        # Load curve library
        lib = cp.Library(path=DX_lib)

        # Define equipment characteristics
        dx_unit = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="si",
            ref_gross_cap=471000,
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
        )

        # Define targeted efficiency
        dx_unit.part_eff = 8.5

        # Define the base curves to be use as the starting point in the generation process
        base_curves = cp.SetofCurves()
        base_curves.curves = dx_unit.set_of_curves
        base_curves.eqp = dx_unit

        # Generate the curves
        set_of_curves = dx_unit.generate_set_of_curves(
            base_curves=[base_curves],
            tol=0.01,
            verbose=True,
            vars=["eir-f-t"],
            max_gen=300,
            max_restart=3,
        )

        # Check that all curves have been generated
        assert len(set_of_curves) == 5

    def test_lib_default_props_dx(self):
        assert (
            self.lib.find_set_of_curves_from_lib()[0].eqp.__dict__["part_eff_ref_std"]
            == "ahri_340/360"
        )

    def test_model_type_error(self):
        lib = cp.Library(path=DX_lib)
        with self.assertLogs(level="ERROR") as log:
            with self.assertRaises(ValueError) as cm:
                DX = cp.UnitaryDirectExpansion(
                    compressor_type="scroll",
                    condenser_type="air",
                    compressor_speed="constant",
                    ref_gross_cap=471000,
                    full_eff=5.89,
                    full_eff_unit="cop",
                    part_eff_ref_std="ahri_340/360",
                    model="simplified",
                    sim_engine="energyplus",
                    set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
                )
            self.assertEqual(str(cm.exception), "Model must be 'simplified_bf'")
        self.assertIn("ERROR", log.output[0])
        self.assertIn("Model must be 'simplified_bf'", log.output[0])

    def test_both_capacity_inputs_provided(self):
        lib = cp.Library(path=DX_lib)
        with self.assertLogs(level="ERROR") as log:
            with self.assertRaises(ValueError) as cm:
                DX = cp.UnitaryDirectExpansion(
                    compressor_type="scroll",
                    condenser_type="air",
                    compressor_speed="constant",
                    ref_gross_cap=471000,
                    ref_net_cap=471000,
                    full_eff=5.89,
                    full_eff_unit="cop",
                    part_eff_ref_std="ahri_340/360",
                    model="simplified_bf",
                    sim_engine="energyplus",
                    set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
                )
            self.assertEqual(
                str(cm.exception), "Input must be one and only one capacity input"
            )
        self.assertIn("ERROR", log.output[0])
        self.assertIn("Input must be one and only one capacity input", log.output[0])
