from unittest import TestCase

import copper as cp


class TestChiller(TestCase):
    def test_get_reference_variable(self):
        lib = cp.Library(path="./fixtures/chiller_curves.json")

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
            set_of_curves=lib.get_set_of_curves_by_name(
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
            set_of_curves=lib.get_set_of_curves_by_name(
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
            set_of_curves=lib.get_set_of_curves_by_name(
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
            set_of_curves=lib.get_set_of_curves_by_name(
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
            set_of_curves=lib.get_set_of_curves_by_name(
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
            set_of_curves=lib.get_set_of_curves_by_name(
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
