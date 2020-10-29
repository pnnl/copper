from unittest import TestCase

import copper as cp
import matplotlib.pyplot as plt


class TestCurves(TestCase):
    def test_curves(self):
        lib = cp.Library(path="./fixtures/chiller_curves.json")

        # Curve lookup by name
        c_name = "ElectricEIRChiller_Trane_CVHE_1080kW/7.39COP/Vanes"
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
            ("cond_type", "air"),
            ("source", "EnergyPlus chiller dataset"),
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
        set_of_curves = lib.get_set_of_curves_by_name(c_name)
        self.assertTrue(round(set_of_curves.curves[0].evaluate(6.67, 35), 2) == 0.96)

        # Eport curves
        set_of_curves.name = set_of_curves.name.replace("/", "_")
        set_of_curves.sim_engine = "energyplus"
        self.assertTrue(set_of_curves.export())
