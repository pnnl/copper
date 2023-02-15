from unittest import TestCase
import copper as cp


class TestAlgorithm(TestCase):
    def test_quickstart_guide(self):
        chlr = cp.Chiller(
            ref_cap=300,
            ref_cap_unit="ton",
            full_eff=0.610,
            full_eff_unit="kw/ton",
            part_eff=0.520,
            part_eff_unit="kw/ton",
            sim_engine="energyplus",
            model="ect_lwt",
            compressor_type="screw",
            condenser_type="water",
            compressor_speed="constant",
        )

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-plr"], method="nearest_neighbor", tol=0.005
        )

        full_eff = chlr.calc_rated_eff(eff_type="full")
        part_eff = chlr.calc_rated_eff(eff_type="part")

        self.assertTrue(full_eff < 0.61 * (1 + 0.005), full_eff)
        self.assertTrue(full_eff > 0.61 * (1 - 0.005), full_eff)
        self.assertTrue(part_eff < 0.52 * (1 + 0.005), part_eff)
        self.assertTrue(part_eff > 0.52 * (1 - 0.005), part_eff)

    def test_tutorial_si(self):
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

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-plr"], method="nearest_neighbor", tol=tol
        )

        full_eff = chlr.calc_rated_eff(eff_type="full", unit=chlr.full_eff_unit)
        part_eff = chlr.calc_rated_eff(eff_type="part", unit=chlr.part_eff_unit)

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)

    def test_max_restart(self):
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

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-plr"],
            method="nearest_neighbor",
            tol=tol,
            max_restart=1,
            max_gen=1,
        )

        self.assertTrue(set_of_curves is None)

    def test_random_init(self):
        full_eff_target = 0.650
        part_eff_target = 0.480

        chlr = cp.Chiller(
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

        ga_1 = cp.Generator(chr, random_seed=1)
        ga_1_vals = [ga_1.get_random() for i in range(5)]

        ga_2 = cp.Generator(chr, random_seed=1)
        ga_2_vals = [ga_2.get_random() for i in range(5)]

        ga_3 = cp.Generator(chr)
        ga_3_vals = [ga_3.get_random() for i in range(5)]

        self.assertTrue(ga_1_vals == ga_2_vals)
        self.assertTrue(ga_1_vals != ga_3_vals)

    def test_gradients(self):
        chlr = cp.Chiller(
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

        algo = cp.Generator(equipment=chlr, vars=["eir-f-plr"], method="best_match")
        algo.generate_set_of_curves()

        grad_val = algo.check_gradients()
        self.assertTrue(grad_val == True)

    def test_double_targets(self):
        full_eff_target = 1.188
        full_eff_target_alt = 1.178
        part_eff_target = 0.876
        part_eff_target_alt = 0.869

        chlr = cp.Chiller(
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

        tol = 0.1

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-plr"], method="best_match", tol=tol
        )

        full_eff = chlr.calc_rated_eff(eff_type="full")
        part_eff = chlr.calc_rated_eff(eff_type="part")
        full_eff_alt = chlr.calc_rated_eff(eff_type="full", alt=True)
        part_eff_alt = chlr.calc_rated_eff(eff_type="part", alt=True)

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

        chlr = cp.Chiller(
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

        tol = 0.03

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-t", "eir-f-plr"], method="best_match", tol=tol
        )

        full_eff = chlr.calc_rated_eff(eff_type="full")
        part_eff = chlr.calc_rated_eff(eff_type="part")

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)

    def test_double_targets_lct(self):
        full_eff_target = 1.188
        full_eff_target_alt = 1.178
        part_eff_target = 0.876
        part_eff_target_alt = 0.869

        chlr = cp.Chiller(
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

        tol = 0.1

        set_of_curves = chlr.generate_set_of_curves(
            vars=["eir-f-t", "eir-f-plr"], method="best_match", tol=tol
        )

        full_eff = chlr.calc_rated_eff(eff_type="full")
        part_eff = chlr.calc_rated_eff(eff_type="part")
        full_eff_alt = chlr.calc_rated_eff(eff_type="full", alt=True)
        part_eff_alt = chlr.calc_rated_eff(eff_type="part", alt=True)

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)
        self.assertTrue(full_eff_alt < full_eff_target_alt * (1 + tol), full_eff_alt)
        self.assertTrue(full_eff_alt > full_eff_target_alt * (1 - tol), full_eff_alt)
        self.assertTrue(part_eff_alt < part_eff_target_alt * (1 + tol), part_eff_alt)
        self.assertTrue(part_eff_alt > part_eff_target_alt * (1 - tol), part_eff_alt)

    def test_generator_nn(self):
        # define parameters for chiller class
        primary_std = "ahri_550/590"
        secondary_std = "ahri_551/591"
        out_var = ["eir-f-plr"]

        # specify target efficiencies
        full_eff_target = 0.634
        full_eff_target_alt = 0.634
        part_eff_target = 0.596
        part_eff_target_alt = 0.596

        chlr = cp.Chiller(
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="any",
            ref_cap=225,
            ref_cap_unit="ton",
            full_eff=full_eff_target,
            full_eff_unit="kw/ton",
            full_eff_alt=full_eff_target_alt,
            full_eff_unit_alt="kw/ton",
            part_eff=part_eff_target,
            part_eff_unit="kw/ton",
            part_eff_ref_std=primary_std,
            part_eff_alt=part_eff_target_alt,
            part_eff_unit_alt="kw/ton",
            part_eff_ref_std_alt=secondary_std,
            model="ect_lwt",
            sim_engine="energyplus",
        )

        tol = 0.03
        method = "nearest_neighbor"

        chlr_curves_set = chlr.generate_set_of_curves(
            vars=out_var, tol=tol, method=method
        )

        full_eff = chlr.calc_rated_eff(eff_type="full")
        part_eff = chlr.calc_rated_eff(eff_type="part")
        full_eff_alt = chlr.calc_rated_eff(eff_type="full", alt=True)
        part_eff_alt = chlr.calc_rated_eff(eff_type="part", alt=True)

        self.assertTrue(full_eff < full_eff_target * (1 + tol), full_eff)
        self.assertTrue(full_eff > full_eff_target * (1 - tol), full_eff)
        self.assertTrue(part_eff < part_eff_target * (1 + tol), part_eff)
        self.assertTrue(part_eff > part_eff_target * (1 - tol), part_eff)
        self.assertTrue(full_eff_alt < full_eff_target_alt * (1 + tol), full_eff_alt)
        self.assertTrue(full_eff_alt > full_eff_target_alt * (1 - tol), full_eff_alt)
        self.assertTrue(part_eff_alt < part_eff_target_alt * (1 + tol), part_eff_alt)
        self.assertTrue(part_eff_alt > part_eff_target_alt * (1 - tol), part_eff_alt)
