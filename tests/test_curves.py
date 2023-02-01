from unittest import TestCase

import copper as cp
import matplotlib.pyplot as plt
import numpy as np
import CoolProp.CoolProp as CP
import os

location = os.path.dirname(os.path.realpath(__file__))
chiller_lib = os.path.join(location, "../copper/lib", "chiller_curves.json")


class TestCurves(TestCase):
    def test_curves(self):
        lib = cp.Library(path=chiller_lib)

        # Curve lookup by name
        c_name = "27"
        self.assertTrue(len([lib.get_set_of_curves_by_name(c_name)]))
        with self.assertRaises(ValueError):
            lib.get_set_of_curves_by_name(c_name + "s")

        # Equipment lookup
        self.assertTrue(
            len(lib.find_equipment(filters=[("eqp_type", "chiller")]).keys())
        )
        self.assertFalse(len(lib.find_equipment(filters=[("eqp_type", "vrf")]).keys()))

        # Set of curves lookup using filter
        filters = [
            ("eqp_type", "chiller"),
            ("sim_engine", "energyplus"),
            ("model", "ect_lwt"),
            ("condenser_type", "air"),
            ("source", "2"),
        ]

        set_of_curvess = lib.find_set_of_curvess_from_lib(filters=filters)
        self.assertTrue(len(set_of_curvess) == 111)

        # Plot curves
        out_vars = ["eir-f-t", "cap-f-t", "eir-f-plr"]

        fig, axes = plt.subplots(nrows=1, ncols=len(out_vars), figsize=(25, 5))

        plot_res = []
        for c_set in set_of_curvess:
            plot_res.append(c_set.plot(out_var=out_vars, norm=True, axes=axes))
        self.assertTrue(all(plot_res))

        # Evaluate curve values
        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="variable",
            ref_cap=256800,
            ref_cap_unit="W",
            full_eff=3.355,
            full_eff_unit="cop",
            model="ect_lwt",
            sim_engine="energyplus",
        )
        c_set.eqp = chlr
        set_of_curves = lib.get_set_of_curves_by_name(c_name)
        self.assertTrue(round(set_of_curves.curves[0].evaluate(6.67, 35), 2) == 0.96)

        # Export curves
        set_of_curves.name = set_of_curves.name.replace("/", "_")
        set_of_curves.eqp = chlr
        set_of_curves.sim_engine = "energyplus"
        self.assertTrue(set_of_curves.export())

    def test_curve_conversion(self):
        # Define equipment
        lib = cp.Library(path=chiller_lib, rating_std="ahri_550/590")
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

        # Define curve
        c = cp.Curve(eqp=chlr, c_type="bi_quad")
        c.coeff1 = 0.8205623152958919
        c.coeff2 = 0.015666171280029038
        c.coeff3 = -0.0009618860655775869
        c.coeff4 = 0.00999598253118077
        c.coeff5 = -0.00029391662581783687
        c.coeff6 = 0.0003883447155134793
        c.x_min = 4.0
        c.x_max = 15.6
        c.y_min = 10.0
        c.y_max = 40.0
        c.out_var = "cap-f-t"

        # Convert curves coefficient to IP
        c.convert_coefficients_to_ip()

        # Verify curve coefficients
        assert round(c.coeff1, 3) == 0.090
        assert round(c.coeff2, 3) == 0.024
        assert round(c.coeff3, 3) == 0.000
        assert round(c.coeff4, 3) == 0.008
        assert round(c.coeff5, 3) == -0.000
        assert round(c.coeff6, 3) == 0.000

        assert c.x_min == 39.2
        assert c.y_min == 50.0
        assert c.x_max == 60.08
        assert c.y_max == 104

    def test_agg(self):
        filters = [
            ("eqp_type", "chiller"),
            ("sim_engine", "energyplus"),
            ("model", "ect_lwt"),
            ("condenser_type", "water"),
        ]

        # Run unittest with a centrifugal chiller
        lib = cp.Library(path=chiller_lib)
        ep_wtr_screw = lib.find_set_of_curvess_from_lib(
            filters=filters + [("source", "2"), ("compressor_type", "screw")]
        )
        ep_wtr_scroll = lib.find_set_of_curvess_from_lib(
            filters=filters + [("source", "2"), ("compressor_type", "scroll")]
        )
        centrifugal_chlr = ep_wtr_screw + ep_wtr_scroll

        sets = centrifugal_chlr
        chlr = cp.Chiller(
            ref_cap=200,
            ref_cap_unit="ton",
            full_eff=5.0,
            full_eff_unit="cop",
            compressor_type="aggregated",
            condenser_type="water",
            compressor_speed="aggregated",
        )
        water_cooled_curves = cp.SetsofCurves(sets=sets, eqp=chlr)
        # remove sets
        for cset in sets:
            cset.remove_curve("eir-f-plr-dt")

        ranges = {
            "eir-f-t": {
                "vars_range": [(4, 10), (10.0, 40.0)],
                "normalization": (6.67, 29.44),
            },
            "cap-f-t": {
                "vars_range": [(4, 10), (10.0, 40.0)],
                "normalization": (6.67, 29.44),
            },
            "eir-f-plr": {"vars_range": [(0.0, 1.0)], "normalization": (1.0)},
        }

        misc_attr = {
            "model": "ect_lwt",
            "ref_cap": 200,
            "ref_cap_unit": "",
            "full_eff": 5.0,
            "full_eff_unit": "",
            "compressor_type": "aggregated",
            "condenser_type": "water",
            "compressor_speed": "aggregated",
            "sim_engine": "energyplus",
            "min_plr": 0.1,
            "min_unloading": 0.1,
            "max_plr": 1,
            "name": "Aggregated set of curves",
            "validity": 3,
            "source": "Aggregation",
        }

        # checking with an empty target, we expect a Value Error
        with self.assertRaises(ValueError):
            _, _ = water_cooled_curves.nearest_neighbor_sort()

        # checking with bad target variables that are not present in library
        with self.assertRaises(AssertionError):
            _, _ = water_cooled_curves.nearest_neighbor_sort(
                target_attr=misc_attr, vars=["bad_targets", "wrong_targets"]
            )

        # first look at the test with weighted average
        df, best_idx = water_cooled_curves.nearest_neighbor_sort(target_attr=misc_attr)
        self.assertEqual(best_idx, 8)  # the best index for this test is 8
        self.assertEqual(np.round(df.loc[best_idx, "score"], 3), 0.068)

        # look at the nearest neighbor-implementation with N nearest neighbor, N=7
        df, best_idx = water_cooled_curves.nearest_neighbor_sort(
            target_attr=misc_attr, N=7
        )
        score = df.loc[[best_idx], ["score"]]["score"].values[0]
        self.assertEqual(best_idx, 8)  # the best index for this test is STILL 8
        self.assertEqual(np.round(score, 3), 0.159)

    def test_flow_calcs_after_agg(self):
        # Load library
        lib = cp.Library(path=chiller_lib)

        # Determine aggregated curve
        ranges = {
            "eir-f-t": {
                "vars_range": [(4, 10), (15.0, 50.0)],
                "normalization": (6.67, 34.44),
            },
            "cap-f-t": {
                "vars_range": [(4, 10), (15.0, 50.0)],
                "normalization": (6.67, 34.44),
            },
            "eir-f-plr": {
                "vars_range": [(15.0, 50.0), (0.0, 1.0)],
                "normalization": (34.44, 1.0),
            },
        }

        misc_attr = {
            "model": "lct_lwt",
            "ref_cap": 100,
            "ref_cap_unit": "",
            "full_eff": 12.0 / 0.72 / 3.412,
            "part_eff": 12.0 / 0.56 / 3.412,
            "ref_eff_unit": "",
            "compressor_type": "aggregated",
            "condenser_type": "water",
            "compressor_speed": "constant",
            "sim_engine": "energyplus",
            "min_plr": 0.1,
            "min_unloading": 0.1,
            "max_plr": 1,
            "name": "Aggregated set of curves",
            "validity": 3,
            "source": "Aggregation",
        }

        # define chiller before passing as argument
        # Define target chiller
        chlr = cp.Chiller(
            compressor_type="scroll, screw, recip",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=150,
            ref_cap_unit="ton",
            full_eff=0.75,
            full_eff_unit="kw/ton",
            full_eff_alt=0.719,
            full_eff_unit_alt="kw/ton",
            part_eff=0.56,
            part_eff_unit="kw/ton",
            part_eff_ref_std="ahri_550/590",
            part_eff_alt=0.559,
            part_eff_unit_alt="kw/ton",
            part_eff_ref_std_alt="ahri_551/591",
            model="lct_lwt",
            sim_engine="energyplus",
        )

        filters = [
            ("eqp_type", "chiller"),
            ("model", "lct_lwt"),
            ("condenser_type", "water"),
            ("sim_engine", "energyplus"),
        ]

        sets = lib.find_set_of_curvess_from_lib(
            filters=filters + [("compressor_type", "centrifugal")], part_eff_flag=True
        )

        curves = cp.SetsofCurves(sets=sets, eqp=chlr)

        base_curve = curves.get_aggregated_set_of_curves(
            ranges=ranges, misc_attr=misc_attr, method="weighted-average", N=10
        )

        base_curve.eqp = chlr

        # Calculate condenser flow
        chlr.set_of_curves = base_curve.curves
        m = chlr.get_ref_cond_flow_rate()

        # Check that the correct condenser flow is calculated
        self.assertTrue(round(m, 3) == 0.030, f"Calculated condenser flow {m} m3/s")

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

        cop = cp.Units(chlr.full_eff, "kw/ton").conversion("cop")

        # Determine the density of water [kg/m3]
        rho = CP.PropsSI(
            "D", "P", 101325, "T", 0.5 * (chlr.ref_ect + chlr.ref_lct) + 273.15, "Water"
        )

        args = [
            chlr.ref_lwt,
            base_curve.curves[1],  # cap-f-t
            base_curve.curves[0],  # eir-f-t
            base_curve.curves[2],  # eir-f-plr
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
