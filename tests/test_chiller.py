from unittest import TestCase

import copper as cp
import pickle as pkl


class TestChiller(TestCase):
    lib = cp.Library(path="./fixtures/chiller_curves.json")

    def test_get_reference_variable(self):

        chlr = cp.chiller(
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
            set_of_curves=self.lib.get_set_of_curves_by_name(
                "ReformEIRChiller_Carrier_19XR_869kW/5.57COP/VSD"
            ).curves,
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

        chlr = cp.chiller(
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
            set_of_curves=self.lib.get_set_of_curves_by_name(
                "ReformEIRChiller_Carrier_19XR_869kW/5.57COP/VSD"
            ).curves,
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

        chlr = cp.chiller(
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
            set_of_curves=self.lib.get_set_of_curves_by_name(
                "ReformEIRChiller_Carrier_19XR_869kW/5.57COP/VSD"
            ).curves,
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
            set_of_curves=self.lib.get_set_of_curves_by_name(
                "ReformEIRChiller_Carrier_19XR_869kW/5.57COP/VSD"
            ).curves,
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

        chlr = cp.chiller(
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
            set_of_curves=self.lib.get_set_of_curves_by_name(
                "ReformEIRChiller_Carrier_19XR_869kW/5.57COP/VSD"
            ).curves,
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

        chlr = cp.chiller(
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
            set_of_curves=self.lib.get_set_of_curves_by_name(
                "ReformEIRChiller_Carrier_19XR_869kW/5.57COP/VSD"
            ).curves,
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

        curves = pkl.load(open("./tests/data/agg_curves.pkl", "rb"))[4]

        chlr = cp.chiller(
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

        self.assertTrue(round(m, 3) == 0.015)

        args = [
            6.67,
            curves[1],
            curves[0],
            curves[2],
            1,
            -999,
            1 / cop,
            29.44,
            m * 1000,
            4.19,
        ]

        lct = chlr.get_lct(29.44, args)

        self.assertTrue(round(lct, 3) == 39.604, f"Calculated LCT: {lct}")
