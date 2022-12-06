Additional Examples
====================

LCT-based chiller performance curves
-------------------------------------

Most building energy modeling software use an entering condenser temperature (ECT) model. Some software tools such as `EnergyPlus`_ can simulate chillers using a leaving condenser temperature (LCT) model. **Copper** can be used to generate performance curves for such a model. The following example generates a set of performance curves for the LCT model for a 100-ton water-cooled scroll chiller.

.. sourcecode:: python

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

    tol = 0.01

    set_of_curves = chlr.generate_set_of_curves(
        vars=["eir-f-t", "eir-f-plr"], method="best_match", tol=tol
    )

Targeting two different rating standards
-----------------------------------------

The rating conditions in AHRI Standards 550/590 and 551/591 are different. **Copper** supports the IPLV and performance curve generation for both standards. It is also possible to generate curves for two sets of targeted efficiencies, one for AHRI 550/590 and the other for 551/591 as demonstrated in the following example:

.. sourcecode:: python

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

    tol = 0.01

    set_of_curves = chlr.generate_set_of_curves(
        vars=["eir-f-plr"], method="best_match", tol=tol
    )

Repeatability
--------------
Because **Copper** is used to find a solution to an underdetermined system of equations, there are a multitude (an infinite) number of solutions to a set of targeted equipment characteristics and efficiencies; hence, running **Copper** multiple times can lead to different sets of curves. For applications where repeatability is necessary, users can use the `random_seed` attribute in an equipment definition, which ensures that the same result is generated every time the same equipment definition is run using **Copper**.

.. sourcecode:: python

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
        random_seed=1
    )

.. _EnergyPlus: https://energyplus.net/