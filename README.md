# Copper
Copper is a performance curve generator for building energy simulations. It uses a genetic algorithm to modify exisiting performance curves to match specific performance ratings at both full and part loads.

Unit Test Status: [![CircleCI](https://circleci.com/gh/lymereJ/copper/tree/master.svg?style=svg&circle-token=2d4c8e2ca7654f95f203d37182a8d2cd4680bbe7)](https://circleci.com/gh/lymereJ/copper/tree/master)

# Equipment and Simulation Software Supported
| Equipment Type | Equipment Subtype | Simulation Software | Algorithm |
| ------------- | ------------- | ------------- | ------------- |
| Chiller | Air-cooled | [EnergyPlus](https://github.com/NREL/EnergyPlus) | Entering condenser and leaving chiller temperature |
| Chiller | Water-cooled | [EnergyPlus](https://github.com/NREL/EnergyPlus) | Entering condenser and leaving chiller temperature |
| Chiller | Air-cooled | [DOE-2](http://www.doe2.com/) | Entering condenser and leaving chiller temperature |
| Chiller | Water-cooled | [DOE-2](http://www.doe2.com/) | Entering condenser and leaving chiller temperature | 
