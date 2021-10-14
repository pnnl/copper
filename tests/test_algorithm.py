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
