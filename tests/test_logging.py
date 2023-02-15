from unittest import TestCase
import copper as cp


class TestLogging(TestCase):
    def test_logging(self):
        full_eff_target = 5.2
        part_eff_target = 7.4

        chlr = cp.Chiller(
            ref_cap=1250,
            ref_cap_unit="kW",
            full_eff=full_eff_target,
            full_eff_unit="cop",
            part_eff=part_eff_target,
            part_eff_unit="cop",
            sim_engine="energyplus",
            model="ect_lwt",
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
        )

        tol = 0.005
        with self.assertLogs() as captured:
            chlr.generate_set_of_curves(
                vars=["eir-f-plr"],
                method="nearest_neighbor",
                tol=tol,
                max_restart=1,
                max_gen=1,
            )
        self.assertTrue(
            captured[0][0].msg
            == "Target not met after 1 generations; Restarting the generator."
        )
        self.assertTrue(captured[0][0].levelname == "WARNING")
        self.assertTrue("GEN: 0, IPLV: 5.56, KW/TON: 5" in captured[0][1].msg)
        self.assertTrue(captured[0][1].levelname == "INFO")
        self.assertTrue(
            captured[0][2].msg
            == "Target not met after 1 restart; No solution was found."
        )
        self.assertTrue(captured[0][2].levelname == "CRITICAL")
