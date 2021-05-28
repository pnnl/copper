from unittest import TestCase
import copper as cp


class TestAlgorithm(TestCase):
    def test_tutorial(self):
        chlr = cp.Chiller(
            ref_cap=300,
            ref_cap_unit="tons",
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
            vars=["eir-f-t", "cap-f-t", "eir-f-plr"], method="typical"
        )

        res = "Efficiency: {} kW/ton, IPLV: {} kW/ton.".format(
            round(chlr.calc_eff(eff_type="kwpton"), 2),
            round(chlr.calc_eff(eff_type="iplv"), 2),
        )

        self.assertTrue(res == "Efficiency: 0.65 kW/ton, IPLV: 0.48 kW/ton.")

    def test_gradients(self):
        chlr = cp.Chiller(
            ref_cap=300,
            ref_cap_unit="tons",
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
        # chlr.generate_set_of_curves(vars=["eir-f-t", "cap-f-t", "eir-f-plr"],
        #                            method="typical")

        algo = cp.GA(equipment=chlr, method="typical")
        algo.generate_set_of_curves()

        grad_val = algo.check_gradients()
        self.assertTrue(grad_val == True)


if __name__ == "__main__":
    algoTest = TestAlgorithm()
    algoTest.test_gradients()
