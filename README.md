# Copper
Copper is a performance curve generator for building energy simulations. It uses a genetic algorithm to modify existing performance curves to match specific performance indicators at both full and part load.

![Tests](https://github.com/lymereJ/copper/actions/workflows/tests.yml/badge.svg)

# Equipment and Simulation Software Supported
| Equipment Type | Equipment Subtype | Simulation Software | Algorithm |
| ------------- | ------------- | ------------- | ------------- |
| Chiller | Air-cooled | [EnergyPlus](https://github.com/NREL/EnergyPlus) | Entering condenser and leaving chiller temperature |
| Chiller | Water-cooled | [EnergyPlus](https://github.com/NREL/EnergyPlus) | Entering condenser and leaving chiller temperature |
| Chiller | Water-cooled | [EnergyPlus](https://github.com/NREL/EnergyPlus) | Leaving condenser and leaving chiller temperature |
| Chiller | Air-cooled | [DOE-2](http://www.doe2.com/) | Entering condenser and leaving chiller temperature |
| Chiller | Water-cooled | [DOE-2](http://www.doe2.com/) | Entering condenser and leaving chiller temperature | 

# References
* EnergyPlus: Energy Simulation Program, 2000, Drury B. Crawley, Curtis O. Pedersen, Linda K. Lawrie, Frederick C. Winkelmann
* DOE-2.2, James J. Hirsch & Associates, http://doe2.com/doe2/
* COMNET, Appendix H – Equipment Performance Curves, https://comnet.org
* 2015 Standard for Performance Unit of Water-chilling and Heat Pump Water-heating Packages Using the Vapor Compression Cycle, 2015, AHRI
* 2016 Nonresidential Alternative Calculation Method, Appendix 5.7, 2015, California Energy Commission
* ANSI/ASHRAE/IES Standard 90.1-2016, 2016, ASHRAE

