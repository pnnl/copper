from unittest import TestCase
import copper as cp


class TestAlgorithm(TestCase):
    def test_tutorial(self):

        full_eff_target = 0.650
        part_eff_target = 0.480

        chlr = cp.chiller(
            ref_cap=300,
            ref_cap_unit="ton",
            full_eff=full_eff_target,
            full_eff_unit="kw/ton",
            part_eff=part_eff_target,
            part_eff_unit="kw/ton",
            sim_engine="energyplus",
            model="ect_lwt",
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
        )

        tol = 0.005

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-plr"], method="best_match", tol=tol
        )

        full_eff = chlr.calc_eff(eff_type="full")
        part_eff = chlr.calc_eff(eff_type="part")

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)

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

        full_eff_target = 1.188
        full_eff_target_alt = 1.178
        part_eff_target = 0.876
        part_eff_target_alt = 0.869

        chlr = cp.chiller(
            compressor_type="scroll",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=100,
            ref_cap_unit="ton",
            full_eff=full_eff_target,
            full_eff_unit="kw/ton",
            full_eff_alt=full_eff_target_alt,
            full_eff_unit_alt="kw/ton",
            part_eff=part_eff_target,
            part_eff_unit="kw/ton",
            part_eff_ref_std="ahri_550/590",
            part_eff_alt=part_eff_target_alt,
            part_eff_unit_alt="kw/ton",
            part_eff_ref_std_alt="ahri_551/591",
            model="ect_lwt",
            sim_engine="energyplus",
        )

        tol = 0.01

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-plr"], method="best_match", tol=tol
        )

        full_eff = chlr.calc_eff(eff_type="full")
        part_eff = chlr.calc_eff(eff_type="part")
        full_eff_alt = chlr.calc_eff(eff_type="full", alt=True)
        part_eff_alt = chlr.calc_eff(eff_type="part", alt=True)

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)
        self.assertTrue(full_eff_alt < full_eff_target_alt * (1 + tol), full_eff_alt)
        self.assertTrue(full_eff_alt > full_eff_target_alt * (1 - tol), full_eff_alt)
        self.assertTrue(part_eff_alt < part_eff_target_alt * (1 + tol), part_eff_alt)
        self.assertTrue(part_eff_alt > part_eff_target_alt * (1 - tol), part_eff_alt)

    def test_single_target_lct(self):

        full_eff_target = 1.188
        part_eff_target = 0.876

        chlr = cp.chiller(
            compressor_type="scroll",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=100,
            ref_cap_unit="ton",
            full_eff=full_eff_target,
            full_eff_unit="kw/ton",
            part_eff=part_eff_target,
            part_eff_unit="kw/ton",
            part_eff_ref_std="ahri_550/590",
            model="lct_lwt",
            sim_engine="energyplus",
        )

        tol = 0.01

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-t", "eir-f-plr"], method="best_match", tol=tol
        )

        full_eff = chlr.calc_eff(eff_type="full")
        part_eff = chlr.calc_eff(eff_type="part")

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)

    def test_double_targets_lct(self):
        full_eff_target = 1.188
        full_eff_target_alt = 1.178
        part_eff_target = 0.876
        part_eff_target_alt = 0.869

        chlr = cp.chiller(
            compressor_type="scroll",
            condenser_type="water",
            compressor_speed="constant",
            ref_cap=100,
            ref_cap_unit="ton",
            full_eff=full_eff_target,
            full_eff_unit="kw/ton",
            full_eff_alt=full_eff_target_alt,
            full_eff_unit_alt="kw/ton",
            part_eff=part_eff_target,
            part_eff_unit="kw/ton",
            part_eff_ref_std="ahri_550/590",
            part_eff_alt=part_eff_target_alt,
            part_eff_unit_alt="kw/ton",
            part_eff_ref_std_alt="ahri_551/591",
            model="lct_lwt",
            sim_engine="energyplus",
        )

        tol = 0.01

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-t", "eir-f-plr"], method="best_match", tol=tol
        )

        full_eff = chlr.calc_eff(eff_type="full")
        part_eff = chlr.calc_eff(eff_type="part")
        full_eff_alt = chlr.calc_eff(eff_type="full", alt=True)
        part_eff_alt = chlr.calc_eff(eff_type="part", alt=True)

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)
        self.assertTrue(full_eff_alt < full_eff_target_alt * (1 + tol), full_eff_alt)
        self.assertTrue(full_eff_alt > full_eff_target_alt * (1 - tol), full_eff_alt)
        self.assertTrue(part_eff_alt < part_eff_target_alt * (1 + tol), part_eff_alt)
        self.assertTrue(part_eff_alt > part_eff_target_alt * (1 - tol), part_eff_alt)
