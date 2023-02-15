# Copper
Copper is a building energy simulation performance curve generator for heating, ventilation, and air-conditioning equipment. It uses a genetic algorithm to modify existing or a set of aggregated existing performance curves to match specific design characteristics including full and part load energy performance metric values.

Copper generates performance curves that can be used in most common building energy software such as [EnergyPlus](https://energyplus.net/) or [DOE-2](https://doe2.com/).

![Tests](https://github.com/lymereJ/copper/actions/workflows/tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7644215.svg)](https://doi.org/10.5281/zenodo.7644215)
[![status](https://joss.theoj.org/papers/9509076e1ef0a0c05f141fde6810c313/status.svg)](https://joss.theoj.org/papers/9509076e1ef0a0c05f141fde6810c313)

# Purpose
Results from building energy simulations largely depend on how heating and cooling equipment are modeled. As they rarely operate at their full load and rated performance it is important that reasonable data is used to model an heating or cooling equipment's part load performance. Copper was created to allow building energy modelers, engineers, and researchers to generate simulation-ready performance curves of heating and cooling equipment that not only capture their typical behavior at part load but are also generated to match a set of design characteristics such as full load and part load efficiencies.

While chillers, which are rated at both full load and part load through an integrated part load value (IPLV), are currently the main application of Copper, other heating and cooling equipment will be handled in the near future.

# Documentation
The documentation for Copper is hosted [here](https://pnnl.github.io/copper/index.html), it includes installation instructions, quick tutorials, examples, API documentation, and more.

# Contributing
Contributions (issues and pull requests) are welcomed and greatly appreciated. Guidelines are provided in the documentation.

# Help
If you need help using Copper, please open an issue and one of the developers will get back to you at their earliest convenience.
