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

    # Define equipment characteristics
    dx_unit_dft = cp.UnitaryDirectExpansion(
        compressor_type="scroll",
        condenser_type="air",
        compressor_speed="constant",
        ref_cap_unit="W",
        ref_gross_cap=471000,
        full_eff=5.89,
        full_eff_unit="cop",
        part_eff_ref_std="ahri_340/360",
        model="simplified_bf",
        sim_engine="energyplus",
        set_of_curves=lib.get_set_of_curves_by_name("D208122216").curves,
    )

    def test_calc_eff_ect(self):
        ieer = round(self.dx_unit_dft.calc_rated_eff(unit="eer"), 1)
        self.assertTrue(7.5 == ieer, f"{ieer} is different than 7.5")

        # Two-speed fan unit
        dx_unit_two_speed = self.dx_unit_dft
        dx_unit_two_speed.indoor_fan_speeds = 2
        ieer_two_spd = round(dx_unit_two_speed.calc_rated_eff(), 2)
        assert ieer_two_spd > ieer

    def test_check_net_gross_capacity(self):
        # Check that the difference between the gross and net capacity is the indoor fan power
        assert round(
            cp.Units(
                value=self.dx_unit_dft.ref_gross_cap - self.dx_unit_dft.ref_net_cap,
                unit=self.dx_unit_dft.ref_cap_unit,
            ).conversion(new_unit="kW"),
            3,
        ) == round(self.dx_unit_dft.indoor_fan_power, 3)

        # Same check but with "ton" based capacity
        dx_unit_alt = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="ton",
            ref_gross_cap=2.5,
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            set_of_curves=self.lib.get_set_of_curves_by_name("D208122216").curves,
        )
        assert round(
            cp.Units(
                value=dx_unit_alt.ref_gross_cap - dx_unit_alt.ref_net_cap,
                unit=dx_unit_alt.ref_cap_unit,
            ).conversion(new_unit="kW"),
            3,
        ) == round(dx_unit_alt.indoor_fan_power, 3)

    def test_check_lib_ieer(self):
        # Get all curves from the library
        filters = [("eqp_type", "UnitaryDirectExpansion")]
        curves = self.lib.find_set_of_curves_from_lib(
            filters=filters, part_eff_flag=True
        )

        for i in range(len(curves)):
            # Get equipment from curves from the library
            eqp = curves[i].eqp

            # Assign a default PLF curve
            plf_f_plr = cp.Curve(eqp=eqp, c_type="linear")
            plf_f_plr.out_var = "plf-f-plr"
            plf_f_plr.type = "linear"
            plf_f_plr.coeff1 = 1 - eqp.degradation_coefficient * 0.9  # TODO: to revise
            plf_f_plr.coeff2 = eqp.degradation_coefficient * 0.9
            plf_f_plr.x_min = 0
            plf_f_plr.x_max = 1
            plf_f_plr.out_min = 0
            plf_f_plr.out_max = 1

            # Re-assigne curve to equipment
            eqp.set_of_curves = curves[i].curves
            eqp.set_of_curves.append(plf_f_plr)

            # Check that the IEER is always better than full load EER
            assert round(eqp.full_eff, 2) < round(
                eqp.calc_rated_eff(eff_type="part", unit="eer"), 3
            )

    def test_multi_speed(self):
        # Two-speed fan unit
        dx_unit_two_speed = self.dx_unit_dft
        dx_unit_two_speed.indoor_fan_speeds = 2
        assert (
            dx_unit_two_speed.calc_fan_power(capacity_ratio=0.5)
            / dx_unit_two_speed.indoor_fan_power
            == 0.4
        )
        assert (
            dx_unit_two_speed.calc_fan_power(capacity_ratio=1.0)
            / dx_unit_two_speed.indoor_fan_power
            == 1.0
        )
        assert (
            dx_unit_two_speed.calc_fan_power(capacity_ratio=0.75)
            / dx_unit_two_speed.indoor_fan_power
            == 0.7
        )

        # Define equipment characteristics
        # Four-speed fan
        dx_unit_four_speed = self.dx_unit_dft
        dx_unit_four_speed.indoor_fan_speeds = 4
        dx_unit_four_speed.indoor_fan_speeds_mapping = {
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
        }
        assert (
            dx_unit_four_speed.calc_fan_power(capacity_ratio=0.1)
            / dx_unit_four_speed.indoor_fan_power
            == 0.15
        )
        assert (
            dx_unit_four_speed.calc_fan_power(capacity_ratio=1.0)
            / dx_unit_four_speed.indoor_fan_power
            == 1.0
        )
        assert (
            dx_unit_four_speed.calc_fan_power(capacity_ratio=0.75)
            / dx_unit_four_speed.indoor_fan_power
            == 0.7
        )
        assert (
            round(
                dx_unit_four_speed.calc_fan_power(capacity_ratio=0.58)
                / dx_unit_four_speed.indoor_fan_power,
                2,
            )
            == 0.53
        )
        assert (
            round(
                dx_unit_four_speed.calc_fan_power(capacity_ratio=0.70)
                / dx_unit_four_speed.indoor_fan_power,
                2,
            )
            == 0.65
        )

    def test_multi_speed_with_curve(self):
        # Two-speed fan unit
        dx_unit_multi_speed = self.dx_unit_dft
        dx_unit_multi_speed.indoor_fan_curve = 1
        dx_unit_multi_speed.indoor_fan_speeds = 2
        assert (
            dx_unit_multi_speed.calc_fan_power(capacity_ratio=0.5)
            / dx_unit_multi_speed.indoor_fan_power
            == 0.25
        )
        assert (
            dx_unit_multi_speed.calc_fan_power(capacity_ratio=1.0)
            / dx_unit_multi_speed.indoor_fan_power
            == 1.0
        )
        assert (
            dx_unit_multi_speed.calc_fan_power(capacity_ratio=0.75)
            / dx_unit_multi_speed.indoor_fan_power
            < 0.7
        )

    def test_generation(self):
        # Define equipment characteristics
        dx_unit = self.dx_unit_dft

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

    def test_generation_best_match(self):
        # Define equipment characteristics
        dx = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="ton",
            ref_gross_cap=8,
            full_eff=11.55,
            full_eff_unit="eer",
            part_eff=14.8,
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
        )
        # Generate the curves
        set_of_curves = dx.generate_set_of_curves(
            tol=0.05,
            verbose=True,
            method="best_match",
            num_nearest_neighbors=5,
            random_seed=1,
            vars=["eir-f-t"],
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

    def test_get_ranges(self):
        ranges = self.dx_unit_dft.get_ranges()
        assert isinstance(ranges, dict)
        assert len(ranges) == 5

    def test_degradation(self):
        self.dx_unit_dft.degradation_coefficient = 0
        self.dx_unit_dft.add_cycling_degradation_curve(overwrite=True)
        assert len(self.dx_unit_dft.set_of_curves) == 5
        assert self.dx_unit_dft.get_dx_curves()["plf-f-plr"].coeff1 == 1.0

    def test_NN_wght_avg(self):
        # Define equipment
        dx = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="ton",
            ref_gross_cap=8,
            full_eff=11.55,
            full_eff_unit="eer",
            part_eff=14.8,
            part_eff_ref_std="ahri_340/360",
            model="simplified_bf",
            sim_engine="energyplus",
            indoor_fan_speeds=2,
            indoor_fan_speeds_mapping={
                "1": {
                    "fan_flow_fraction": 0.66,
                    "fan_power_fraction": 0.4,
                    "capacity_fraction": 0.5,
                },
                "2": {
                    "fan_flow_fraction": 1.0,
                    "fan_power_fraction": 1.0,
                    "capacity_fraction": 1.0,
                },
            },
            indoor_fan_power=cp.Units(value=8, unit="ton").conversion(new_unit="W")
            * 0.05
            / 1000,
            indoor_fan_power_unit="kW",
        )

        # Generate the curves
        set_of_curves = dx.generate_set_of_curves(
            method="nearest_neighbor",
            tol=0.005,
            num_nearest_neighbors=5,
            verbose=True,
            vars=["eir-f-t", "plf_f_plr"],
            random_seed=1,
        )

        # Check that all curves have been generated
        assert len(set_of_curves) == 5

        # Check normalization
        assert round(set_of_curves[0].evaluate(19.44, 35), 2) == 1.0
        assert round(set_of_curves[1].evaluate(19.44, 35), 2) == 1.0
        assert round(set_of_curves[2].evaluate(1.0, 0), 2) == 1.0
        assert round(set_of_curves[3].evaluate(1.0, 0), 2) == 1.0
        assert round(set_of_curves[4].evaluate(1.0, 0), 2) == 1.0


# Run the tests
import unittest

if __name__ == "__main__":
    unittest.main()
