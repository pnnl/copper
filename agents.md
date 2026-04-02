# Copper - AI Agent Documentation

## Project Overview

**Copper** is a building energy simulation performance curve generator for HVAC (heating, ventilation, and air-conditioning) equipment. It uses genetic algorithms to modify existing or aggregated performance curves to match specific design characteristics including full and part load energy performance metrics.

- **Primary Language**: Python (3.10+)
- **Package Manager**: Poetry (also supports pip via requirements.txt)
- **License**: Open source (see LICENSE.md)
- **Organization**: Pacific Northwest National Laboratory (PNNL)
- **Purpose**: Generate simulation-ready performance curves for building energy modeling software (EnergyPlus, DOE-2)

## Project Structure

```
copper/
├── copper/                      # Main package directory
│   ├── __init__.py             # Package initialization, imports main classes
│   ├── chiller.py              # Chiller equipment class and calculations
│   ├── unitarydirectexpansion.py  # Unitary DX equipment (air conditioners, heat pumps)
│   ├── equipment.py            # Base equipment class
│   ├── generator.py            # Genetic algorithm curve generator
│   ├── curves.py               # Performance curve classes and operations
│   ├── library.py              # Curve library management
│   ├── schema.py               # JSON schema validation
│   ├── units.py                # Unit conversion utilities
│   ├── helpers.py              # Helper functions
│   ├── i_o_process.py          # Input/output processing
│   ├── cli.py                  # Command-line interface
│   ├── constants.py            # Project constants
│   └── data/                   # JSON libraries of performance curves
├── tests/                       # Unit tests
├── docs/                        # Documentation source
├── build/                       # Built documentation (HTML)
├── applications/                # Application examples
├── dev/                         # Development notebooks and scripts
├── pyproject.toml              # Poetry configuration
├── requirements.txt            # Pip dependencies
└── setup.py                    # Package setup
```

## Core Concepts

### 1. Equipment Types
- **Chiller**: Water-cooled or air-cooled chillers with various compressor types
- **UnitaryDirectExpansion**: Packaged air conditioners and heat pumps

### 2. Performance Curves
Performance curves are mathematical representations of equipment efficiency at various operating conditions:
- **cap-f-t**: Capacity as a function of temperature
- **eir-f-t**: Energy Input Ratio as a function of temperature
- **eir-f-plr**: Energy Input Ratio as a function of Part Load Ratio
- **cap-f-ff**: Capacity as a function of flow fraction
- **eir-f-ff**: EIR as a function of flow fraction
- **plf-f-plr**: Part Load Fraction as a function of Part Load Ratio

### 3. Curve Types
- **linear**: 2 coefficients
- **quad** (quadratic): 3 coefficients
- **cubic**: 4 coefficients
- **bi_quad** (bi-quadratic): 6 coefficients
- **bi_cub** (bi-cubic): 10 coefficients

### 4. Generation Methods
- **best_match**: Find the single best matching curve from library
- **nearest_neighbor**: Aggregate N nearest neighbors with weighted averaging
- **weighted_average**: Weighted average of all library curves
- **genetic_algorithm**: Optimize curve coefficients to match performance targets

## Key Classes

### Equipment Hierarchy
```
Equipment (base class)
├── Chiller
└── UnitaryDirectExpansion
```

### Main Classes
- **Equipment**: Base class for all HVAC equipment
- **Chiller**: Chiller-specific calculations and curve generation
- **UnitaryDirectExpansion**: DX equipment calculations
- **Generator**: Genetic algorithm implementation for curve optimization
- **Curve**: Individual performance curve representation
- **SetofCurves**: Collection of curves for one piece of equipment
- **SetsofCurves**: Multiple sets for aggregation
- **Library**: Manages curve libraries (JSON files)
- **Schema**: Validates input JSON files

## Naming Conventions

### Variables
- **ref_cap**: Reference capacity
- **full_eff**: Full load efficiency
- **part_eff**: Part load efficiency (IPLV for chillers, IEER for DX)
- **eir**: Energy Input Ratio (inverse of COP)
- **cop**: Coefficient of Performance
- **plr**: Part Load Ratio
- **lwt**: Leaving Water Temperature
- **ect**: Entering Condenser Temperature
- **lct**: Leaving Condenser Temperature
- **aew**: Air Entering Wet-bulb
- **aed**: Air Entering Dry-bulb

### Efficiency Units
- **kW/ton**: Kilowatts per ton of cooling
- **cop**: Coefficient of Performance
- **eer**: Energy Efficiency Ratio
- **ieer**: Integrated Energy Efficiency Ratio
- **iplv**: Integrated Part Load Value

### File Naming
- Test files: `test_*.py`
- Data files: `*_curves.json` for curve libraries
- Development files in `dev/`: notebooks (`.ipynb`), scripts (`.py`), data files

## Common Workflows

### 1. Generate Chiller Curves
```python
from copper import Chiller, Generator

# Define chiller
chiller = Chiller(
    ref_cap=500,                    # Reference capacity
    ref_cap_unit="ton",
    full_eff=0.56,                  # Full load efficiency
    full_eff_unit="kW/ton",
    part_eff=0.52,                  # Part load efficiency (IPLV)
    part_eff_unit="kW/ton",
    compressor_type="centrifugal",
    condenser_type="water",
    compressor_speed="variable"
)

# Generate curves
gen = Generator(chiller, method="nearest_neighbor")
gen.generate_set_of_curves()
```

### 2. Generate DX Equipment Curves
```python
from copper import UnitaryDirectExpansion, Generator

# Define equipment
dx = UnitaryDirectExpansion(
    ref_net_cap=35000,              # Net cooling capacity in Watts
    ref_cap_unit="W",
    full_eff=3.5,                   # Full load COP
    full_eff_unit="cop",
    part_eff=4.2,                   # IEER
    part_eff_unit="eer",
    compressor_type="scroll"
)

# Generate curves
gen = Generator(dx, method="weighted_average")
gen.generate_set_of_curves()
```

### 3. Using CLI
```bash
copper run input.json
```

## Important Implementation Details

### Genetic Algorithm Parameters
- **pop_size**: Population size (default: 100)
- **max_gen**: Maximum generations (default: 300)
- **tol**: Tolerance for target matching (default: 0.005)
- **retain**: Fraction of best performers to retain (default: 0.2)
- **mutate**: Mutation probability (default: 0.95)
- **random_select**: Random selection probability (default: 0.1)

### Chiller Models
- **ect_lwt**: DOE-2 model (Entering Condenser Temp, Leaving Water Temp)
- **lct_lwt**: Reformulated EIR model (Leaving Condenser Temp, Leaving Water Temp)

### DX Models
- **simplified_bf**: Simplified bypass factor model (currently only supported model)

### Rating Standards
- **ahri_550/590**: Air-cooled and water-cooled chillers
- **ahri_340/360**: Unitary air conditioners and heat pumps
- **ashrae_90.1**: ASHRAE Standard 90.1 compliance

## Dependencies

### Core
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **scipy**: Optimization algorithms
- **statsmodels**: Statistical modeling
- **matplotlib**: Plotting

### Domain-Specific
- **CoolProp**: Thermodynamic properties of fluids
- **click**: CLI framework
- **jsonschema**: JSON validation

## Testing

- Tests located in `tests/` directory
- Run tests: `pytest tests/`
- Test coverage: Use `coverage` package
- Each module has corresponding test file (e.g., `chiller.py` → `test_chiller.py`)

## Input/Output

### Input Formats
- **JSON**: Primary input format for CLI and programmatic use
- **Python objects**: Direct instantiation of equipment classes

### Output Formats
- **JSON**: Curve coefficients and metadata
- **CSV**: Tabular curve data
- **IDF**: EnergyPlus Input Data File format
- **Excel**: Formatted reports

## Error Handling

- Extensive validation of input parameters
- Schema validation for JSON inputs
- Logging throughout (`logging` module)
- Assertions for critical parameters (compressor type, model type, etc.)

## Conventions for AI Agents

### When Modifying Code
1. **Maintain backward compatibility** - This is a library used by others
2. **Add tests** for new functionality in `tests/`
3. **Update documentation** if adding new features
4. **Follow existing patterns** - Look at similar equipment types for guidance
5. **Validate inputs** - Equipment parameters must be physically reasonable
6. **Use logging** - Don't print, use `logging.info()`, `logging.error()`, etc.

### When Adding New Equipment
1. Inherit from `Equipment` base class
2. Implement required methods: `calc_rated_eff()`, `get_rated_temperatures()`, `get_lib_and_filters()`
3. Define curve types in `get_ranges()`
4. Add library file in `copper/data/`
5. Update `equipment_references.json` with rating standards

### When Debugging
1. Check `logging` output - extensive logging throughout
2. Validate JSON inputs with Schema
3. Check curve normalization at reference conditions
4. Verify unit conversions (use `Units` class)
5. Check genetic algorithm convergence (increase `max_gen` if needed)

### Common Pitfalls
- **Units**: Always use the `Units` class for conversions, never manual conversion
- **Reference conditions**: Curves must normalize to 1.0 at reference conditions
- **COP vs EIR**: Remember EIR = 1/COP
- **Gross vs Net capacity**: DX equipment has both (difference is fan power)
- **Curve coefficients**: Use 1-based indexing (coeff1, coeff2, etc.)

## Configuration Files

- **pyproject.toml**: Poetry configuration, defines package metadata and dependencies
- **setup.py**: Fallback setup for pip installation
- **requirements.txt**: Pip-compatible dependency list

## Key Algorithms

### Genetic Algorithm Flow
1. Generate initial population by randomly modifying base curves
2. Calculate fitness (distance from target efficiency)
3. Select best performers
4. Crossover and mutation to create new generation
5. Repeat until target met or max generations reached

### Curve Aggregation
1. Find similar equipment in library using filters
2. Calculate distance/similarity metrics
3. Weight curves by similarity
4. Aggregate coefficients using weighted average

## Documentation

- **Source**: `docs/` directory (reStructuredText)
- **Built**: `build/` directory (HTML)
- **Hosted**: https://pnnl.github.io/copper/
- **Includes**: API docs, tutorials, examples, methodology

## Web Application

- **URL**: https://copper.pnnl.gov/
- Provides GUI access to Copper functionality

---

**Last Updated**: 2026-04-02
**For Questions**: Open an issue on GitHub or see documentation at https://pnnl.github.io/copper/
