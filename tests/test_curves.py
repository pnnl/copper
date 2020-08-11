from unittest import TestCase

import copper as cp
import matplotlib.pyplot as plt


class TestCurves(TestCase):
    def test_curves(self):
        lib = cp.Library(path="./fixtures/chiller_curves.json")

        # Curve lookup by name
        c_name = "ElectricEIRChiller_Trane_CVHE_1080kW/7.39COP/Vanes"
        self.assertTrue(len([lib.get_curve_set_by_name(c_name)]))
        with self.assertRaises(ValueError):
            lib.get_curve_set_by_name(c_name + "s")

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
        out_vars = ["eir-f-t", "cap-f-t", "eir-f-plr"]

        fig, axes = plt.subplots(nrows=1, ncols=len(out_vars), figsize=(25, 5))

        plot_res = []
        for c_set in curve_sets:
            plot_res.append(
                c_set.plot(out_var=out_vars, norm=True, axes=axes)
            )
        self.assertTrue(all(plot_res))

        # Evaluate curve values
        curve_set = lib.get_curve_set_by_name(c_name)
        self.assertTrue(round(curve_set.curves[0].evaluate(6.67, 35), 2) == 0.96)
