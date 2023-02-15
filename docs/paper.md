---
title: 'Copper: a performance curve generator for building energy simulation'
tags:
  - python
  - energy
  - building
  - simulation
  - hvac
authors:
  - name: Jérémy Lerond
    orcid: 0000-0002-1630-6886
    affiliation: 1
  - name: Aowabin Rahman
    orcid: 0000-0003-2543-5126
    affiliation: 1
  - name: Jian Zhang
    orcid: 0000-0001-6125-0407
    affiliation: 1
  - name: Michael Rosenberg
    orcid: 0000-0003-1490-3115
    affiliation: 1
affiliations:
 - name: Pacific Northwest National Laboratory, Richland, WA, USA
   index: 1
date: 22 September 2022
bibliography: paper.bib
---

# Summary

For decades, building energy simulation has been used by architects, engineers, and researchers to evaluate the performance of building designs. Heating, ventilation, and air-conditioning systems are one of the main energy end-users in buildings; hence, it is important to reasonably capture the performance of this equipment in simulations at rated (as defined, for instance, in AHRI and ASHRAE Standards [@ahri_550590; @ahri_551591; @90.1]), full load, and part load conditions. `Copper` is a performance curve generator created to enable building energy simulation practitioners to generate simulation-ready performance curves for heating and cooling equipment that not only capture the equipment's typical behavior at part load, but also match a set of design characteristics, including full load and part load efficiencies.

# Statement of need

While performance curves for heating and cooling equipment can be derived from manufacturers' data (assuming enough data can be gathered), building energy situationists often require the use of generic performance curves to simulate the energy consumption of heating and cooling equipment in buildings. In such instances, one has to rely on publicly available performance curve repositories such as COMNET [@comnet], which in many cases contain curves that are not specific enough -- that is, sets of curves that are not metric specific (i.e., do not match specific rated efficiency values, especially part load values) or are used to represent too broad a range of equipment size. `Copper` was created to solve these challenges by generating generic sets of performance curves for a specific set of design characteristics using a data-driven approach [@ashrae_conf_paper]. The goal is to provide a quick, reliable, and versatile tool to help building energy modeling practitioners using software such as EnergyPlus [@osti_1395882] or DOE-2 [@doe2] with early-design analyzes, building energy code development and compliance, and more broadly any other simulation-based building research project or analysis needing such performance curves.


`Copper` has been used to support research and development efforts for building energy code compliance methodologies. It was also used to generate sets of minimally code compliant chiller performance curves to be used normatively in ASHRAE Standard 90.1 [@addendumbd] and has been integrated in Washington State's performance-based energy code compliance tool for HVAC systems [@tspr].

# Acknowledgements

Copper was developed at the Pacific Northwest National Laboratory, and was partially funded under contract with the U.S. Department of Energy (DOE). It is actively being developed as an open-source project.

# References
