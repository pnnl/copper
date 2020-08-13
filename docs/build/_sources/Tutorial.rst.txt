Tutorial
=========

Chiller
---------
This section shows how to go about using `Copper` to generate chiller performance curves.

First, let's start by importing necessary packages:

.. sourcecode:: python

    import copper as cp
    import matplotlib.pyplot as plt

Second, define the chiller that you wish to create a curve for. We wish to generate curves for a 300 ton chiller with an efficiency of 0.650 kW/ton and an IPLV of 0.480 kW/ton.

.. sourcecode:: python

    chlr = cp.Chiller(ref_cap=300, ref_cap_unit="tons",
                    full_eff=0.650, full_eff_unit="kw/ton",
                    part_eff=0.48, part_eff_unit="kw/ton",
                    sim_engine="energyplus",
                    model="ect_lwt",
                    compressor_type="centrifugal", 
                    condenser_type="water",
                    compressor_speed="constant")

Then, generate a set of curves for it.

.. sourcecode:: python

    chlr.generate_set_of_curves(vars=['eir-f-t','cap-f-t','eir-f-plr'],
                                method="typical", sFac=0.9, 
                                tol=0.005, random_select=0.3, mutate=0.8)

Finally, plot the curves.

.. sourcecode:: python

    # Define plot and variables to plot
    out_vars = ['eir-f-t', 'cap-f-t', 'eir-f-plr']
    fig, axes = plt.subplots(nrows=1, ncols=len(out_vars), figsize=(25,5))
    
    # Plotting space set of curves
    new_curves = cp.SetofCurves("chiller")
    new_curves.curves = chlr.set_of_curves
    new_curves.plot(out_var=out_vars, 
                    axes=axes, 
                    color='darkolivegreen', 
                    alpha=1)

This should produce something like the following figure.

.. image:: chiller_curves.png

Let's check that the set of curves would result in simulation a chiller with an efficiency of 0.650 kW/ton and an IPLV of 0.480 kW/ton

.. sourcecode:: python

    print("Efficiency: {} kW/ton, IPLV: {} kW/ton.".format(round(chlr.calc_eff(eff_type="kwpton"),2), 
                                                           round(chlr.calc_eff(eff_type="iplv"),2)))

This will return `Efficiency: 0.65 kW/ton, IPLV: 0.48 kW/ton.`
Once this is done you can also export the set of curves to the simulation engine input format.

.. sourcecode:: python

    new_curves.export(path="./curves/curve")

You do not need to include the extension when passing the `path` argument.