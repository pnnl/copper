Additional Examples
====================

LCT-based Chiller Performance Curves
-------------------------------------

Most building energy modeling software use an entering condenser temperature (ECT) model. Some software such as `EnergyPlus`_ have capabilities to simulate chillers using a model using a leaving condenser temperature (LCT) model. **Copper** can be used to generate performance curves for such a model. The following example generate a set of performance curves for the LCT model for a 100 ton water cooled scroll chiller.

.. sourcecode:: python

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

Targeting Two Different Rating Standards
-----------------------------------------

The rating conditions in AHRI Standard 550/590 and 551/591 are different. **Copper** support the IPLV and performance curve generation for both standards. It is possible to also generate curves for two sets of targeted efficiencies, one for AHRI 550/590, and the other one for 551/591. The following example demonstrate that.

.. sourcecode:: python

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

.. _EnergyPlus: https://energyplus.net/