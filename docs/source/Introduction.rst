Introduction
=============
Results from building energy simulations largely depend on how heating and cooling equipment is modeled. Because heating and cooling equipment rarely operate at its full load and rated performance it is important that reasonable data be used to model the equipment’s part load performance. **Copper** was created to enable building energy modelers, engineers, and researchers to generate simulation-ready performance curves of heating and cooling equipment that not only capture the equipment's typical behavior at part load operation, but also match a set of design characteristics such as full load and part load efficiencies.

Performance curves are multi-variate polynomial regression models describing the performance of given equipment across a range of operating conditions. Typically, part load performance of heating and cooling equipment is simulated using one or multiple, or a set of, performance curves. For instance, chiller models most often use three different part load ratios whereas boiler models tend to use one performance curve.

What can **Copper** do?
------------------------
* Generate curves
* Plot curves
* Export curves

What equipment does **Copper** handle?
--------------------------------------------
* Chillers

Future development
-------------------
* Support for packaged rooftop direct expansion equipment
* Generation of ASHRAE Standard 205 compliant equipment performance maps
