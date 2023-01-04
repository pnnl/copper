Additional Examples
====================

LCT-based chiller performance curves
-------------------------------------

Most building energy modeling software use an entering condenser temperature (ECT) model. Some software tools such as `EnergyPlus`_ can simulate chillers using a leaving condenser temperature (LCT) model as documented in the `engineering manual`_. **Copper** can be used to generate performance curves for such a model. The following example generates a set of performance curves for the LCT model for a 100-ton water-cooled scroll chiller.

.. sourcecode:: python

    import copper as cp

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

    import copper as cp

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

    import copper as cp

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
        compressor_speed="constant"
    )

    set_of_curves = chlr.generate_set_of_curves(
        vars=["eir-f-plr"], method="nearest_neighbor", tol=0.005, random_seed=1
    )

.. _EnergyPlus: https://energyplus.net/
.. _engineering manual: https://bigladdersoftware.com/epx/docs/22-2/engineering-reference/chillers.html#electric-chiller-model-based-on-condenser-leaving-temperature