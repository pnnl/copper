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
            ("condenser_type", "air"),
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
        chlr = cp.chiller(
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
        lib = cp.Library(path="./fixtures/chiller_curves.json", rating_std = "ahri_550/590")
        chlr = cp.chiller(
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
            set_of_curves=lib.get_set_of_curves_by_name("ElectricEIRChiller_McQuay_WSC_471kW/5.89COP/Vanes").curves
        )

        # Define curve
        c = cp.Curve(eqp=chlr, c_type='bi_quad')
        c.coeff1=0.8205623152958919
        c.coeff2=0.015666171280029038
        c.coeff3=-0.0009618860655775869
        c.coeff4=0.00999598253118077
        c.coeff5=-0.00029391662581783687
        c.coeff6=0.0003883447155134793
        c.x_min=4.0
        c.x_max=15.6
        c.y_min=10.0
        c.y_max=40.0
        c.out_var='cap-f-t'

        # Convert curves coefficient to IP
        c.convert_coefficients_to_ip()

        # Verify curve coefficients
        assert(c.coeff1 == 0.09018668973148625)
        assert(c.coeff2 == 0.023868143705118694)
        assert(c.coeff3 == -0.0002968784153017188)
        assert(c.coeff4 == 0.007523580775319991)
        assert(c.coeff5 == -9.071500796847256e-05)
        assert(c.coeff6 == 0.00011985948009674967)

        assert(c.x_min == 39.2)
        assert(c.y_min == 50.0)
        assert(c.x_max == 60.08)
        assert(c.y_max == 104)
