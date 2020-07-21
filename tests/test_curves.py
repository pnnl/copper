from unittest import TestCase

import copper as cp
import matplotlib.pyplot as plt


class TestCurves(TestCase):
    def test_curves(self):
        lib = cp.Library(path="./fixtures/chiller_curves.json")

        # Curve lookup by name
        c_name = "ElectricEIRChiller_Trane_CVHE_1080kW/7.39COP/Vanes"
        self.assertTrue(len([lib.get_curve_set_by_name(c_name, eqp_match="chiller")]))
        with self.assertRaises(ValueError):
            lib.get_curve_set_by_name(c_name + "s", eqp_match="chiller")

        # Equipment lookup
        self.assertTrue(
            len(lib.find_equipment(filters=[("eqp_type", "chiller")]).keys())
        )
        self.assertFalse(len(lib.find_equipment(filters=[("eqp_type", "vrf")]).keys()))

        # Curve set lookup using filter
        filters = [
            ("eqp_type", "chiller"),
            ("sim_engine", "energyplus"),
            ("model", "ect_lwt"),
            ("cond_type", "air"),
            ("source", "EnergyPlus chiller dataset"),
        ]

        curve_sets = lib.find_curve_sets_from_lib(filters=filters)
        self.assertTrue(len(curve_sets) == 111)

        # Plot curves
        ranges = {
            "eir-f-t": {
                "x1_min": 6.67,
                "x1_max": 6.67,
                "x1_norm": 6.67,
                "nbval": 50,
                "x2_min": 20,
                "x2_max": 30,
                "x2_norm": 30,
            },
            "cap-f-t": {
                "x1_min": 6.67,
                "x1_max": 6.67,
                "x1_norm": 6.67,
                "nbval": 50,
                "x2_min": 20,
                "x2_max": 30,
                "x2_norm": 30,
            },
            "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": 50},
            "eir-f-plr-dt": {
                "x1_min": 0.3,
                "x1_max": 1,
                "x1_norm": 1,
                "nbval": 50,
                "x2_min": 23.33,
                "x2_max": 23.33,
                "x2_norm": 23.33,
            },
        }

        out_vars = ["eir-f-t", "cap-f-t", "eir-f-plr"]

        fig, axes = plt.subplots(nrows=1, ncols=len(out_vars), figsize=(25, 5))

        plot_res = []
        for c_set in curve_sets:
            plot_res.append(
                c_set.plot(out_var=out_vars, ranges=ranges, norm=True, axes=axes)
            )
        self.assertTrue(all(plot_res))

        # Evaluate curve values
        curve_set = lib.get_curve_set_by_name(c_name, eqp_match="chiller")
        self.assertTrue(round(curve_set.curves[0].evaluate(6.67, 35), 2) == 0.96)
