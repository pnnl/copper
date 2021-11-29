from unittest import TestCase
import copper as cp


class TestAlgorithm(TestCase):
    def test_tutorial(self):
        chlr = cp.chiller(
            ref_cap=300,
            ref_cap_unit="ton",
            full_eff=0.650,
            full_eff_unit="kw/ton",
            part_eff=0.48,
            part_eff_unit="kw/ton",
            sim_engine="energyplus",
            model="ect_lwt",
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
        )

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-plr"], method="best_match", tol=0.005
        )

        res = "Efficiency: {} kW/ton, IPLV: {} kW/ton.".format(
            round(chlr.calc_eff(eff_type="full"), 2),
            round(chlr.calc_eff(eff_type="part"), 2),
        )

        self.assertTrue(res == "Efficiency: 0.65 kW/ton, IPLV: 0.48 kW/ton.")

    def test_gradients(self):
        chlr = cp.chiller(
            ref_cap=300,
            ref_cap_unit="ton",
            full_eff=0.650,
            full_eff_unit="kw/ton",
            part_eff=0.48,
            part_eff_unit="kw/ton",
            sim_engine="energyplus",
            model="ect_lwt",
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
        )

        algo = cp.GA(equipment=chlr, vars=["eir-f-plr"], method="best_match")
        algo.generate_set_of_curves()

        grad_val = algo.check_gradients()
        self.assertTrue(grad_val == True)
    
    def test_double_targets(self):
        chlr = cp.chiller(
            compressor_type="scroll",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=100,
            ref_cap_unit="ton",
            full_eff=1.188,
            full_eff_unit="kw/ton",
            full_eff_alt=1.178,
            full_eff_unit_alt="kw/ton",
            part_eff=0.876,
            part_eff_unit="kw/ton",
            part_eff_ref_std="ahri_550/590",
            part_eff_alt=0.869,
            part_eff_unit_alt="kw/ton",
            part_eff_ref_std_alt="ahri_551/591",
            model="ect_lwt",
            sim_engine="energyplus"
        )

        set_of_curves = chlr.generate_set_of_curves(vars=["eir-f-plr"], method="best_match", tol=0.005)

        res = "Efficiency: {} kW/ton, IPLV: {} kW/ton, Alt-Efficiency: {} kW/ton, Alt-IPLV: {}.".format(
            round(chlr.calc_eff(eff_type="full"), 2),
            round(chlr.calc_eff(eff_type="part"), 2),
            round(chlr.calc_eff(eff_type="full", alt=True), 2),
            round(chlr.calc_eff(eff_type="part", alt=True), 2),
        )

        self.assertTrue(res == "Efficiency: 1.18 kW/ton, IPLV: 0.88 kW/ton, Alt-Efficiency: 1.18 kW/ton, Alt-IPLV: 0.87.")

    def test_double_targets_lct(self):
        chlr = cp.chiller(
            compressor_type="scroll",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=100,
            ref_cap_unit="ton",
            full_eff=1.188,
            full_eff_unit="kw/ton",
            full_eff_alt=1.178,
            full_eff_unit_alt="kw/ton",
            part_eff=0.876,
            part_eff_unit="kw/ton",
            part_eff_ref_std="ahri_550/590",
            part_eff_alt=0.869,
            part_eff_unit_alt="kw/ton",
            part_eff_ref_std_alt="ahri_551/591",
            model="lct_lwt",
            sim_engine="energyplus"
        )

        set_of_curves = chlr.generate_set_of_curves(vars=["eir-f-plr"], method="best_match", tol=0.005)

        res = "Efficiency: {} kW/ton, IPLV: {} kW/ton, Alt-Efficiency: {} kW/ton, Alt-IPLV: {}.".format(
            round(chlr.calc_eff(eff_type="full"), 2),
            round(chlr.calc_eff(eff_type="part"), 2),
            round(chlr.calc_eff(eff_type="full", alt=True), 2),
            round(chlr.calc_eff(eff_type="part", alt=True), 2),
        )

        self.assertTrue(res == "Efficiency: 1.18 kW/ton, IPLV: 0.88 kW/ton, Alt-Efficiency: 1.19 kW/ton, Alt-IPLV: 0.87.")