"""
copper.py
====================================
This is the core module of Copper. It handles the following:

- Curves
- Sets of curves
- Equipment related calculations
- Unit conversions
- Genetic algorithm
- Curve library manipulations
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json, copy, random, statistics, itertools


import criteria


class Library:
    def __init__(self, path="~/PycarmProjects/codes/copper/fixtures/chiller_curves.json"): #./fixtures/chiller_curves.json"

        self.path = path
        self.data = json.loads(open(self.path, "r").read())


    def content(self):
        return self.data

    def get_unique_eqp_fields(self):
        """Get all unique values for each field of a particular equipment.

        :return: Dictionary showing all unique values for each equipment field.
        :rtype: dict[str, ]

        """
        # Store all value for each field
        uni_field_val = {}
        for _, eqp_f in self.data.items():
            for field, val in eqp_f.items():
                if field != "curves" and field != "name":
                    # Check if field has already been added
                    if field not in uni_field_val.keys():
                        uni_field_val[field] = [val]
                    else:
                        uni_field_val[field].append(val)
        # Retain only unique values
        for field, val in uni_field_val.items():
            uni_field_val[field] = set(val)
        return uni_field_val

    def find_set_of_curvess_from_lib(self, filters=[]):
        """Retrieve sets of curves from a library matching specific filters.

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :return: All set of curves object matching the filters
        :rtype: list()

        """
        # Find name of equiment that match specified filter
        eqp_match = self.find_equipment(filters)

        # List of sets of curves that match specified filters
        set_of_curvess = []

        # Retrieve identified equipment's sets of curves from the library
        for name, props in eqp_match.items():
            c_set = SetofCurves(props["eqp_type"])

            # Retrive all attributes of the sets of curves object
            for c_att in list(c_set.__dict__):
                # Set the attribute of new Curve object
                # if attrubute are identified in database entry
                if c_att in list(self.data[name].keys()):
                    c_set.__dict__[c_att] = self.data[name][c_att]

            c_lst = []

            # Create new SetofCurves and Curve objects for all the
            # sets of curves identified as matching the filters
            for c in self.data[name]["curves"]:
                c_lst.append(
                    self.get_curve(
                        c, self.data[name], eqp_type=self.data[name]["eqp_type"]
                    )
                )
            c_set.curves = c_lst
            set_of_curvess.append(c_set)

        return set_of_curvess

    def find_equipment(self, filters=[]):
        """Find equipment matching specified filter in the curve library.
        
        Special filter characters:
        
        - ~! means "all except..."
        - ! means "do not include..."
        - ~ means "include..."

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :return: Dictionary of field for each equipment matching specified filter 
        :rtype: dict[str,dict[]]

        """
        eqp_match_dict = {}
        for eqp in self.data:
            assertions = []
            for prop, val in filters:
                # ~! = all but...
                if "~!" in val:
                    assertions.append(
                        val.replace("~!", "").lower().strip()
                        not in self.data[eqp][prop].lower().strip()
                    )
                # ! = does not include
                elif "!" in val:
                    assertions.append(self.data[eqp][prop] != val)
                # ~ = includes
                elif "~" in val:
                    assertions.append(
                        val.replace("~", "").lower().strip()
                        in self.data[eqp][prop].lower().strip()
                    )
                else:
                    assertions.append(self.data[eqp][prop] == val)
            if all(assertions):
                eqp_match_dict[eqp] = self.data[eqp]

        return eqp_match_dict

    def get_set_of_curves_by_name(self, name):
        """Retrieve set of curves from the library by name.

        :param str name: Curve name
        :return: Set of curves object
        :rtype: SetofCurves()

        """
        # Initialize set of curves object
        c_set = SetofCurves("chiller")
        c_set.name = name

        # List of curves
        c_lst = []
        # Define curve objects
        try:
            for c in self.data[name]["curves"]:
                c_lst.append(
                    self.get_curve(
                        c, self.data[name], eqp_type=self.data[name]["eqp_type"]
                    )
                )
            # Add curves to set of curves object
            c_set.curves = c_lst
            return c_set
        except:
            raise ValueError("Cannot find curve in library.")

    def get_curve(self, c, c_name, eqp_type):
        """Retrieve individual attribute of a curve object.

        :param Curve() c: Curve object
        :param str c_name: Name of the curve object
        :param str eqp_type: Type of equipment associated with the curve
        :return: Curve object
        :rtype: Curve()

        """
        # Curve properties
        c_prop = c_name["curves"][c]
        # Initialize curve object
        c_obj = Curve(eqp_type, c_prop["type"])
        c_obj.out_var = c
        # Retrive all attributes of the curve object
        for c_att in list(Curve(eqp_type, c_prop["type"]).__dict__):
            # Set the attribute of new Curve object
            # if attrubute are identified in database entry
            if c_att in list(c_prop.keys()):
                c_obj.__dict__[c_att] = c_prop[c_att]
        return c_obj

    def find_base_curves(self, filters, eqp):
        """Find an existing equipment curve that best matches the equipment.

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :param eqp: Equipment object(e.g. Chiller())
        :return: Set of curves object
        :rtype: SetofCurves()

        """
        # Find equipment match in the library
        eqp_match = self.find_equipment(filters=filters)

        if len(eqp_match) > 0:
            # If multiple equipment match the specified properties,
            # return the one that has numeric attributes that best
            # match the proposed case
            if len(eqp_match) > 1:
                return self.get_set_of_curves_by_name(
                    self.get_best_match(eqp, eqp_match)
                )
            else:
                return self.get_set_of_curves_by_name(eqp_match)
        else:
            raise ValueError(
                "Could not find a set of curves that matches the specified properties."
            )

    def get_best_match(self, eqp, matches):
        """Find the set of curves matching the equipment characteristics the best.

        :param eqp: Equipment object(e.g. Chiller())
        :param dict[str,dict[]] matches: All potential matches
        :return: Name of the set of curves that best matches the equipment characteristics
        :rtype: str

        """
        # Initialize numeric attribute difference
        diff = float("inf")

        # Iterate over matches and calculate the
        # difference in numeric fields
        for name, val in matches.items():
            # Retrieve full load/reference numeric attribute
            if eqp.type == "chiller":
                cap = val["ref_cap"]
                cap_unit = val["ref_cap_unit"]
                eff = val["ref_eff"]
                eff_unit = matches[name]["ref_eff_unit"]

                if not cap is None:
                    # Capacity conversion
                    if cap_unit != eqp.ref_cap_unit:
                        c_unit = Unit(cap, cap_unit)
                        cap = c_unit.conversion(eqp.ref_cap_unit)
                        cap_unit = eqp.ref_cap_unit

                    # Efficiency conversion
                    if eff_unit != eqp.full_eff_unit:
                        c_unit = Unit(eff, eff_unit)
                        eff = c_unit.conversion(eqp.full_eff_unit)
                        eff_unit = eqp.full_eff_unit

                        # Compute difference
                        c_diff = abs((eqp.ref_cap - cap) / eqp.ref_cap) + abs(
                            (eqp.full_eff - eff) / eqp.full_eff
                        )

                        if c_diff < diff:
                            # Update lowest numeric difference
                            diff = c_diff

                            # Update best match
                            best_match = name

        return best_match


class Chiller:
    def __init__(
        self,
        ref_cap,
        ref_cap_unit,
        full_eff,
        full_eff_unit,
        part_eff,
        part_eff_unit,
        compressor_type,
        condenser_type,
        compressor_speed,
        set_of_curves="",
        model="ect_lwt",
        sim_engine="energyplus",
    ):
        self.type = "chiller"
        self.compressor_type = compressor_type
        self.condenser_type = condenser_type
        self.compressor_speed = compressor_speed
        self.ref_cap = ref_cap
        self.ref_cap_unit = ref_cap_unit
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.model = model
        self.sim_engine = sim_engine
        self.set_of_curves = set_of_curves

    def generate_set_of_curves(
        self,
        method="typical",
        pop_size=100,
        tol=0.005,
        max_gen=15000,
        vars="",
        sFac=0.5,
        retain=0.2,
        random_select=0.1,
        mutate=0.95,
        bounds=(6, 10),
        base_curves=[],
    ):
        """Generate a set of curves for a particular Chiller() object.

        :param str method: Method used to generate the set of curves, either `typical` or `best_match`

                           - `typical` uses typical curves and modify them to reach a particular IPLV
                           - `best_match` uses curves that best match the chiller object description
        :param int pop_size: Population size used by the genetic algorithm
        :param float tol: Tolerance used by the genetic algorithm to determine if the proposed solution is acceptable
                          The lower, the more stringent
        :param int max_gen: Maximum number of generation
        :param list() vars: List of variable to run the alorithm on
        :param float sFac: Linear fitness normalization factor, the higher the more aggressive the normalization will be
        :param float retain: Probability of retaining an individual in the next generation
        :param float random_select: Probability of randomly selecting an individual to be part of the next generation
        :param float mutate: Probability of an individual to be mutated in the next generation
        :param tuple() bounds: Random modification bounds (TODO: add more details)
        :return: Set of curves object generated by the genetic algorithm that matches the Chiller() definition
        :rtype: SetofCurves()

        """
        ga = GA(
            self,
            method,
            pop_size,
            tol,
            max_gen,
            vars,
            sFac,
            retain,
            random_select,
            mutate,
            bounds,
            base_curves,
        )
        return ga.generate_set_of_curves()

    def calc_eff(self, eff_type, unit="kw/ton"):
        """Calculate chiller efficiency.

        :param str eff_type: Chiller performance indicator, currently supported `kw/ton` (full load rating) 
                             and `iplv` (part load rating)
        :param str unit: Unit of the efficiency indicator
        :return: Chiller performance indicator
        :rtype: float

        """
        # Retrieve equipment efficiency and unit
        kwpton_ref = self.full_eff
        kwpton_ref_unit = self.full_eff_unit

        # Convert to kWpton if necessary
        if self.full_eff_unit != "kw/ton":
            kwpton_ref_unit = Unit(kwpton_ref, kwpton_ref_unit)
            kwpton_ref = kwpton_ref_unit.conversion("kw/ton")

        # Conversion factors
        ton_to_kbtu = 12
        kbtu_to_kw = 3.412141633

        # Full load conditions
        load_ref = 1
        eir_ref = 1 / (ton_to_kbtu / kwpton_ref / kbtu_to_kw)

        # Test conditions
        # Same approach as EnergyPlus
        # Same as AHRI Std 550/590
        loads = [1, 0.75, 0.5, 0.25]

        # List of equipment efficiency for each load
        kwpton_lst = []

        # DOE-2 chiller model
        if self.model == "ect_lwt":
            if self.condenser_type == "air":
                # Temperatures from AHRI Std 550/590
                chw = 6.67
                ect = [3 + 32 * loads[0], 3 + 32 * loads[1], 3 + 32 * loads[2], 13]
            elif self.condenser_type == "water":
                # Temperatures from AHRI Std 550/590
                chw = 6.67
                ect = [8 + 22 * loads[0], 8 + 22 * loads[1], 19, 19]

            # Retrieve curves
            for curve in self.set_of_curves:
                if curve.out_var == "cap-f-t":
                    cap_f_t = curve
                elif curve.out_var == "eir-f-t":
                    eir_f_t = curve
                else:
                    eir_f_plr = curve

            # Calculate EIR for each testing conditions
            try:
                for idx, load in enumerate(loads):
                    dt = ect[idx] - chw
                    cap_f_chw_ect = cap_f_t.evaluate(chw, ect[idx])
                    eir_f_chw_ect = eir_f_t.evaluate(chw, ect[idx])
                    cap_op = load_ref * cap_f_chw_ect
                    plr = (
                        load * cap_f_t.evaluate(chw, ect[0]) / cap_op
                    )  # Pending EnergyPlus development team review otherwise load / cap_op
                    eir_plr = eir_f_plr.evaluate(plr, dt)
                    # eir = power / load so eir * plr = (power / load) * (load / cap_op)
                    eir = eir_ref * eir_f_chw_ect * eir_plr / plr
                    kwpton = eir / kbtu_to_kw * ton_to_kbtu
                    if eff_type == "kwpton" and idx == 0:
                        return kwpton
                    kwpton_lst.append(eir / kbtu_to_kw * ton_to_kbtu)

                # Coefficients from AHRI Std 550/590
                iplv = 1 / (
                    (0.01 / kwpton_lst[0])
                    + (0.42 / kwpton_lst[1])
                    + (0.45 / kwpton_lst[2])
                    + (0.12 / kwpton_lst[3])
                )
            except:
                return -999
        else:
            # TODO: implement IPLV calcs for other chiller algorithm
            raise ValueError("Algorithm not implemented.")

        # Convert IPLV to desired unit
        if unit != "kw/ton":
            iplv_org = Unit(iplv, "kw/ton")
            iplv = iplv_org.conversion(unit)

        return iplv


class SetsofCurves:
    def __init__(self, eqp_type, sets):
        self.eqp_type = eqp_type
        self.sets_of_curves = sets

    def get_aggregated_set_of_curves(self, method="average", ranges={}, misc_attr={}):
        """
        Determine sets of curves based on aggregation.

        :param string method: Type of aggregation, currently supported: 'average' and 'median.
        :param dict ranges: Dictionary that defines the ranges of values for each independent variable used to calculate aggregated dependent variable values.
        :param dict misc_attr: Dictionary that provides values for the aggregated set of curves.
        :return: Aggregated set of curves.
        :rtype: SetofCurves()
        """
        # Check that all curves are/have:
        # - the same output variables
        # - of the same type
        # - defined for the same type of units

        'first entry -> refsetofcurves'
        ref_setofcurves = self.sets_of_curves[0].list_to_dict()

        for set_of_curves in self.sets_of_curves:
            if set(ref_setofcurves.keys()) != set_of_curves.list_to_dict().keys():
                raise ValueError(
                    "The output variables in each set of curves are not consistently the same, aggregated set of curves cannot currently be determined."
                )
            for c in set_of_curves.curves:
                if c.type != ref_setofcurves[c.out_var].type:
                    raise ValueError(
                        "Curve type in each set of curves are not consistently the same, aggregated set of curves cannot currently be determined."
                    )
                if not c.out_var in list(ranges.keys()):
                    raise ValueError(
                        "Ranges provided do not cover some of the output variables. Ranges: {}, output variables not found in ranges {}.".format(
                            list(ranges.keys()), c.out_var
                        )
                    )
                if c.units != ref_setofcurves[c.out_var].units:
                    raise ValueError(
                        "Curve unit mismatch, aggregated set of curves cannot currently be determined."
                    )

        input_values = {}
        # Determine values of independent variables using the user-specified ranges
        for c in self.sets_of_curves[0].curves:
            input_values[c.out_var] = []
            for vars_rng in ranges[c.out_var]["vars_range"]:
                min_val, max_val = vars_rng
                input_values[c.out_var].append(np.linspace(min_val, max_val, 4))
            # Add 0s for second independent variables for univariate curves
            if len(input_values[c.out_var]) == 1:
                input_values[c.out_var].append(np.linspace(0.0, 0.0, 4))

        output_values = {}
        # Calculate values of dependent variables using the user-specified ranges
        for set_of_curves in self.sets_of_curves:
            for c in set_of_curves.curves:
                output_value = [
                    c.evaluate(x, y)
                    for x, y in list(
                        itertools.product(
                            input_values[c.out_var][0], input_values[c.out_var][1]
                        )
                    )
                ]
                if c.out_var in output_values.keys():
                    output_values[c.out_var].append(output_value)
                else:
                    output_values[c.out_var] = []
                    output_values[c.out_var].append(output_value)

        # Create new set of curves and assign its attributes based on user defined inputs
        agg_set_of_curves = SetofCurves(eqp_type=self.eqp_type)
        for att, att_val in misc_attr.items():
            setattr(agg_set_of_curves, att, att_val)

        # Determine aggregated values for dependent variables
        for var, vals in output_values.items():
            if method == "average":
                y_s = [list(map(lambda x: sum(x) / len(x), zip(*vals)))]
            elif method == "median":
                y_s = [list(map(lambda x: statistics.median(x), zip(*vals)))]

            data = pd.DataFrame(
                [
                    list(xs + (y,))
                    for xs, y in zip(
                        list(
                            itertools.product(
                                input_values[var][0], input_values[var][1]
                            )
                        ),
                        y_s[0],
                    )
                ]
            )
            data.columns = ["X1", "X2", "Y"]

            # Create new curve
            new_curve = Curve(eqp_type=self.eqp_type, c_type=ref_setofcurves[var].type)

            # Assign curve attributes, assume no min/max
            # TODO: Allow min/max to be passed by user
            new_curve.out_var = var
            new_curve.units = self.sets_of_curves[0].curves[0].units
            new_curve.x_min = -999
            new_curve.y_min = -999
            new_curve.x_max = 999
            new_curve.y_max = 999
            new_curve.out_min = -999
            new_curve.out_max = 999
            if not "normalization" in ranges[var].keys():
                raise ValueError(
                    "Normalization point not provided, the curve cannot be created."
                )
            norm = ranges[var]["normalization"]
            if isinstance(norm, float):
                ref_x = norm
                ref_y = 0
            else:
                ref_x, ref_y = norm
            new_curve.ref_x = ref_x
            new_curve.ref_y = ref_y
            # TODO: update fields below when adding new equipment
            if self.eqp_type == "chiller":
                self.ref_evap_fluid_flow = 0
                self.ref_cond_fluid_flow = 0
                if agg_set_of_curves.model == "ect_lwt":
                    self.ref_lwt = ref_y
                    self.ref_ect = ref_x
                else:
                    raise ValueError("Algorithm not supported.")

            # Find curves coefficients
            new_curve.regression(data)

            # Normalize curve to reference point
            new_curve.normalized(data, ref_x, ref_y)

            agg_set_of_curves.curves.append(new_curve)
        return agg_set_of_curves


class SetofCurves:
    def __init__(self, eqp_type):
        self.name = ""
        self.curves = []
        self.validity = 1.0
        self.source = ""
        self.eqp_type = eqp_type
        self.model = ""
        if eqp_type == "chiller":
            self.ref_cap = 0
            self.ref_cap_unit = ""
            self.ref_eff = 0
            self.ref_eff_unit = ""
            self.comp_type = ""
            self.cond_type = ""
            self.comp_speed = ""
            self.sim_engine = ""
            self.min_plr = 0
            self.min_unloading = 0
            self.max_plr = 0
            if self.cond_type == "water":
                self.plotting_range = {
                    "eir-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 35,
                    },
                    "cap-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 35,
                    },
                    "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": 50},
                    "eir-f-plr-dt": {
                        "x1_min": 0.3,
                        "x1_max": 1,
                        "x1_norm": 1,
                        "nbval": 50,
                        "x2_min": 28.33,
                        "x2_max": 28.33,
                        "x2_norm": 28.33,
                    },
                }
            else:
                self.plotting_range = {
                    "eir-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 29.44,
                    },
                    "cap-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 29.44,
                    },
                    "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": 50},
                    "eir-f-plr-dt": {
                        "x1_min": 0.3,
                        "x1_max": 1,
                        "x1_norm": 1,
                        "nbval": 50,
                        "x2_min": 22.77,
                        "x2_max": 22.77,
                        "x2_norm": 22.77,
                    },
                }



    def get_data_for_plotting(self, curve, norm):
        var = curve.out_var
        nb_vals = self.plotting_range[var]["nbval"]
        x1_min = self.plotting_range[var]["x1_min"]
        x1_max = self.plotting_range[var]["x1_max"]
        x_1_vals = np.linspace(x1_min, x1_max, nb_vals)

        if "x2_min" in self.plotting_range[var].keys():
            x2_min = self.plotting_range[var]["x2_min"]
            x2_max = self.plotting_range[var]["x2_max"]
            x_2_vals = np.linspace(x2_min, x2_max, nb_vals)
        else:
            x_2_vals = [0]

        y = []
        for v in range(nb_vals):
            if "x2_min" in self.plotting_range[var].keys():
                norm_fac = (
                    curve.evaluate(
                        self.plotting_range[var]["x1_norm"],
                        self.plotting_range[var]["x2_norm"],
                    )
                    if norm
                    else 1
                )
                y_val = curve.evaluate(x_1_vals[v], x_2_vals[v]) / norm_fac
            else:
                norm_fac = (
                    curve.evaluate(
                        self.plotting_range[var]["x1_norm"],
                        self.plotting_range[var]["x1_norm"],
                    )
                    if norm
                    else 1
                )
                y_val = curve.evaluate(x_1_vals[v], x_1_vals[v]) / norm_fac
            y.append(y_val)

        x = x_1_vals if len(set(x_1_vals)) > len(set(x_2_vals)) else x_2_vals

        return [x, y]

    def plot(self, out_var=[], axes=[], norm=True, color="Black", alpha=0.3):
        """Plot set of curves.

        :param list() out_var: List of the output variables to plot, e.g. `eir-f-t`, `eir-f-plr`, `cap-f-t`.
                               Refer to JSON files structure for other output variables
        :param matplotlib.pyplot.axes axes: Matplotlib pyplot axes
        :param boolean norm: Normalize plot to reference values
        :param str color: Set of curves color
        :param float alpha: Transparency of the curves (0-1).
        :return: Plotting success
        :rtype: boolean

        """
        for i, var in enumerate(out_var):
            for curve in self.curves:
                if curve.out_var == var:
                    x, y = self.get_data_for_plotting(curve, norm)
                    axes[i].plot(x, y, color=color, alpha=alpha)
                    axes[i].set_title(var)
                    #plt.show()

        return True

    def export(self, path="./curves", fmt="idf"):
        """Export curves to simulation engine input format.

        :param str path: Path and file name, do not include the extension,
                         it will be added based on the simulation engine 
                         of the SetofCurves() object.
        :param str fmt: Input format type, currently not used. TODO: json, idf, inp.
        :return: Success
        :rtype: boolean

        """
        curve_export = ""
        for curve in self.curves:
            curve_type = curve.type
            if self.sim_engine == "energyplus":
                if curve_type == "quad":
                    cuvre_type = "Curve:Quadratic"
                elif curve_type == "bi_quad":
                    cuvre_type = "Curve:Biquadratic"
                elif curve_type == "bi_cub":
                    cuvre_type = "Curve:Bicubic"
                curve_export += (
                    "\n{},\n".format(cuvre_type)
                    if len(curve_export)
                    else "{},\n".format(cuvre_type)
                )
                curve_export += "   {}_{},\n".format(self.name, curve.out_var)
                for i in range(1, curve.nb_coeffs() + 1):
                    curve_export += "   {},\n".format(
                        getattr(curve, "coeff{}".format(i))
                    )
                curve_export += (
                    "   {},\n".format(curve.x_min) if curve.x_min else "    ,\n"
                )
                curve_export += (
                    "   {},\n".format(curve.x_max) if curve.x_max else "    ,\n"
                )
                if curve_type != "quad":
                    curve_export += (
                        "   {},\n".format(curve.y_min) if curve.y_min else "    ,\n"
                    )
                    curve_export += (
                        "   {},\n".format(curve.y_max) if curve.y_max else "    ,\n"
                    )
                curve_export += (
                    "   {},\n".format(curve.out_min) if curve.out_min else "    ,\n"
                )
                curve_export += (
                    "   {};\n".format(curve.out_max) if curve.out_max else "    ;\n"
                )
            else:
                # TODO: implement export to DOE-2 format
                raise ValueError(
                    "Export to the {} input format is not yet implemented.".format(
                        self.sim_engine
                    )
                )
        filen = open(path + self.name + ".{}".format(fmt), "w+")
        filen.write(curve_export)
        return True

    def remove_curve(self, out_var):
        """
        Remove curve for a particular output variables from the set

        :param string out_var: Name of the output variable to remove from the set.

        """
        curves_to_del = []
        for c in self.curves:
            if c.out_var == out_var:
                curves_to_del.append(c)
        for c in self.curves:
            if c in curves_to_del:
                self.curves.remove(c)

    def list_to_dict(self):
        """
        Convert curves from the set from a list to a dictionary, the key being the output variable type.

        :return: Dictionary of curves
        :rtype: dict
        """
        curves = {}
        for c in self.curves:
            curves[c.out_var] = c
        return curves


class Curve:
    def __init__(self, eqp_type, c_type):
        # General charactersitics
        self.out_var = ""
        self.type = c_type
        self.units = "si"
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.out_min = 0
        self.out_max = 0
        self.ref_x = 0
        self.ref_y = 0
        if self.type == "quad":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
        elif self.type == "bi_quad":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
            self.coeff4 = 0
            self.coeff5 = 0
            self.coeff6 = 0
        elif self.type == "bi_cub":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
            self.coeff4 = 0
            self.coeff5 = 0
            self.coeff6 = 0
            self.coeff7 = 0
            self.coeff8 = 0
            self.coeff9 = 0
            self.coeff10 = 0

        # Equipment specific charcatertics
        if eqp_type == "chiller":
            self.ref_evap_fluid_flow = 0
            self.ref_cond_fluid_flow = 0
            self.ref_lwt = 6.67
            self.ref_ect = 29.4
            self.ref_lct = 35

    def evaluate(self, x, y):
        """Return the output of a curve.

        :param float x: First curve independent variable
        :param float y: Second curve independent variable
        :return: Curve output
        :rtype: float

        """
        # Catch nulls
        if self.out_min is None:
            self.out_min = -999
        if self.out_max is None:
            self.out_max = 999
        if self.x_min is None:
            self.x_min = -999
        if self.x_max is None:
            self.x_max = 999
        if self.y_min is None:
            self.y_min = -999
        if self.y_max is None:
            self.y_max = 999

        x = min(max(x, self.x_min), self.x_max)
        y = min(max(y, self.y_min), self.y_max)

        if self.type == "bi_quad":
            out = (
                self.coeff1
                + self.coeff2 * x
                + self.coeff3 * x ** 2
                + self.coeff4 * y
                + self.coeff5 * y ** 2
                + self.coeff6 * x * y
            )
            return min(max(out, self.out_min), self.out_max)
        if self.type == "bi_cub":
            out = (
                self.coeff1
                + self.coeff2 * x
                + self.coeff3 * x ** 2
                + self.coeff4 * y
                + self.coeff5 * y ** 2
                + self.coeff6 * x * y
                + self.coeff7 * x ** 3
                + self.coeff8 * y ** 3
                + self.coeff9 * y * x ** 2
                + self.coeff10 * x * y ** 2
            )
            return min(max(out, self.out_min), self.out_max)
        if self.type == "quad":
            out = self.coeff1 + self.coeff2 * x + self.coeff3 * x ** 2
            return min(max(out, self.out_min), self.out_max)

    def nb_coeffs(self):
        """Find number of curve coefficients.

        :return: Number of curve coefficients
        :rtype: int

        """
        ids = []
        for key in list(self.__dict__.keys()):
            if "coeff" in key:
                ids.append(int(key.split("coeff")[-1]))
        return max(ids)

    def regression(self, data):
        """Find curve coefficient by running a multivariate linear regression.

        :param DataFrame() data: DataFrame() object with the following columns: "X1","X1^2","X2","X2^2","X1*X2", "Y"

        """
        # TODO: implement bi_cubic
        if self.type == "quad":
            data["X1^2"] = data["X1"] * data["X1"]
            X = data[["X1", "X1^2"]]
            y = data["Y"]

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            self.coeff1, self.coeff2, self.coeff3 = model.params
        elif self.type == "bi_quad":
            data["X1^2"] = data["X1"] * data["X1"]
            data["X2^2"] = data["X2"] * data["X2"]
            data["X1*X2"] = data["X1"] * data["X2"]

            X = data[["X1", "X1^2", "X2", "X2^2", "X1*X2"]]
            y = data["Y"]

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            self.coeff1, self.coeff2, self.coeff3, self.coeff4, self.coeff5, self.coeff6 = (
                model.params
            )
            r_sqr = model.rsquared
            if r_sqr < 0.8:
                print(
                    "Performance of the regression for {} is poor, r2: {}".format(
                        self.out_var, round(r_sqr, 2)
                    )
                )

    def get_out_reference(self):
        if "-t" in self.out_var:
            x_ref = self.ref_lwt
            y_ref = self.ref_ect
        else:
            x_ref = 1
            y_ref = 0
        return self.evaluate(x_ref, y_ref)

    def normalized(self, data, x_norm, y_norm):
        """Normalize curve around the reference data points.

        :param float x_norm: First independent variable normalization points.
        :param float y_norm: Second independent variable normalization points.
        :param DataFrame() data: DataFrame() object with the following columns: "X1","X1^2","X2","X2^2","X1*X2", "Y".

        """
        data["Y"] = data.apply(
            lambda row: self.evaluate(row["X1"], row["X2"])
            / self.evaluate(x_norm, y_norm),
            axis=1,
        )
        self.regression(data)


class Unit:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_unit):
        """Convert efficiency rating.

        :param str new_unit: Unit after conversion
        :return: Converted value
        :rtype: int

        """
        if new_unit == "kw/ton":
            if self.unit == "cop":
                return 12.0 / (self.value * 3.412)
            elif self.unit == "kw/ton":
                return self.value
            elif self.unit == "EER":
                return 12.0 / self.value
            else:
                return self.value
        elif new_unit == "cop":
            if self.unit == "kw/ton":
                return 12.0 / self.value / 3.412
            elif self.unit == "cop":
                return self.value
            elif self.unit == "EER":
                return self.value / 3.412
            else:
                return self.value
        elif new_unit == "EER":
            if self.unit == "kw/ton":
                return 12.0 / self.value
            elif self.unit == "EER":
                return self.value
            elif self.unit == "cop":
                return 3.412 * self.value
            else:
                return self.value
        elif new_unit == "ton":
            if self.unit == "kW":
                return self.value * (3412 / 12000)
        elif new_unit == "kW":
            if self.unit == "ton":
                return self.value / (3412 / 12000)


class GA:
    def __init__(
        self,
        equipment,
        method="typical",
        pop_size=100,
        tol=0.005,
        max_gen=15000,
        vars="",
        sFac=0.5,
        retain=0.2,
        random_select=0.1,
        mutate=0.95,
        bounds=(6, 10),
        base_curves=[],
    ):
        self.equipment = equipment #chiller
        self.method = method
        self.pop_size = pop_size
        self.tol = tol
        self.max_gen = max_gen
        self.vars = vars
        self.sFac = sFac
        self.retain = retain
        self.random_select = random_select
        self.mutate = mutate
        self.bounds = bounds
        self.base_curves = base_curves

    def generate_set_of_curves(self):
        """Generate set of curves using genetic algorithm.

        :return: Set of curves
        :rtype: SetofCurves()

        """
        self.target = self.equipment.part_eff
        self.full_eff = self.equipment.full_eff

        # Convert target if different than kw/ton
        if self.equipment.part_eff_unit != "kw/ton":
            target_c = Unit(self.target, self.equipment.part_eff_unit)
            self.target = target_c.conversion("kw/ton")

        # Convert target if different than kw/ton
        if self.equipment.full_eff_unit != "kw/ton":
            full_eff_c = Unit(self.equipment.full_eff, self.equipment.full_eff_unit)
            self.full_eff = full_eff_c.conversion("kw/ton")

        if len(self.base_curves) == 0:
            if self.equipment.type == "chiller":
                # TODO: implement other methods
                if self.method == "typical":
                    lib = Library(path="./fixtures/typical_curves.json")
                    #lib = Library(path="./fixtures/typical_curves.json")
                elif self.method == "best_match":
                    lib = Library(path="./fixtures/chiller_curves.json")

                # Define chiller properties
                filters = [
                    ("eqp_type", self.equipment.type),
                    ("comp_type", self.equipment.compressor_type),
                    ("cond_type", self.equipment.condenser_type),
                    ("comp_speed", self.equipment.compressor_speed),
                    ("sim_engine", self.equipment.sim_engine),
                    ("model", self.equipment.model),
                ]
            else:
                raise ValueError("This type of equipment has not yet been implemented.")

            # Find typical curves from library
            # Only one equipment should be returned
            if self.method == "typical":
                self.base_curves = lib.find_set_of_curvess_from_lib(filters)
            elif self.method == "best_match":
                self.base_curves = [lib.find_base_curves(filters, self.equipment)]

        self.set_of_base_curves = self.base_curves[0]
        self.base_curves_data = {}
        for curve in self.set_of_base_curves.curves:
            self.base_curves_data[
                curve.out_var
            ] = self.set_of_base_curves.get_data_for_plotting(curve, False)

        # Run GA
        self.run_ga(curves=self.base_curves)
        return self.equipment.set_of_curves

    def run_ga(self, curves):
        """Run genetic algorithm.

        :param SetofCurves() curves: Initial set of curves to be modified by the algorithm
        :return: Final population of sets of curves
        :rtype: list()

        """
        self.pop = self.generate_population(curves)


        gen = 0
        self.equipment.curves = curves
        while gen <= self.max_gen and not self.is_target_met():
            self.evolve_population(self.pop)
            gen += 1
            # For debugging
            # print("GEN: {}, IPLV: {}, KW/TON: {}".format(gen, round(self.equipment.calc_eff(eff_type="iplv"),2), round(self.equipment.calc_eff(eff_type="kwpton"),2)))
        print("Curve coefficients calculated in {} generations.".format(gen))
        return self.pop

    def is_target_met(self):
        """Check if the objective of the optimization through the algorithm have been met.

        :return: Verification result
        :rtype: boolean

        """
        'Troubleshoot'


        if self.equipment.type == "chiller":
            if self.equipment.set_of_curves != "":
                part_rating = self.equipment.calc_eff(eff_type="iplv")
                full_rating = self.equipment.calc_eff(eff_type="kwpton")
                cap_rating = 0
                if "cap-f-t" in self.vars:
                    for c in self.equipment.set_of_curves: #list of objects  # c in curves
                        #set_of_curves
                        if "cap" in c.out_var:
                            cap_rating += abs(1 - c.get_out_reference())
            else:
                return False
        else:
            raise ValueError("This type of equipment has not yet been implemented.")


        if (
            (part_rating < self.target * (1 + self.tol))
            and (part_rating > self.target * (1 - self.tol))
            and (full_rating < self.full_eff * (1 + self.tol))
            and (full_rating > self.full_eff * (1 - self.tol))
            and (cap_rating < self.tol)
            and (cap_rating > -self.tol)
            and self.check_gradients()
        ):
            return True
        else:
            return False



    'Aowabins function'
    def check_gradients(self):

        'This function checks the gradient and sees if they are monotonic'
        if self.equipment.type == "chiller":
            if self.equipment.set_of_curves != "":

                grad_list = []
                for c in self.equipment.set_of_curves:
                    if c.out_var == 'eir-f-t' or c.out_var == 'eir-f-plr':
                        sign_val = +1
                    elif c.out_var == 'cap-f-t':
                        sign_val = -1
                    else:
                        raise ValueError('this curve output has not been implemented')

                    #to DO
                    x, y = self.set_of_base_curves.get_data_for_plotting(c, False)
                    grad_list.append(self.compute_grad(x=x, y=y, sign_val=sign_val))

                if np.all(np.asarray(grad_list)):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False



    def compute_grad(self, x, y, sign_val):

        grad = np.gradient(y, x)
        sign = np.sign(grad)

        if np.any(sign != -sign_val): #include 0 and +1/-1 gradients. but not gradients of the opposite sign
            return True
        else:
            return False

    def generate_population(self, curves):
        """Generate population of sets of curves.

        :param SetofCurves() curves: Initial set of curves to be modified by the algorithm
        :return: Verification result
        :rtype: boolean

        """
        pop = []
        for _ in range(self.pop_size):
            pop.append(self.individual(curves))
        return pop

    def get_random(self):
        """Generate random number between two bounds.

        :return: Randomly generated value
        :rtype: float

        """
        while True:
            val = float(
                random.randrange(-99999, 99999)
                / 10 ** (random.randint(self.bounds[0], self.bounds[1]))
            )
            if val != 0:
                return val

    def individual(self, curves):
        """Create new individual.

        :param SetofCurves() curves: Initial set of curves to be modified by the algorithm
        :return: New set of curves randomly modified
        :rtype: SetofCurves

        """
        new_curves = copy.deepcopy(curves[0])
        for curve in new_curves.curves:
            if len(self.vars) == 0 or curve.out_var in self.vars:
                for idx in range(1, 11):
                    try:
                        setattr(
                            curve,
                            "coeff{}".format(idx),
                            getattr(curve, "coeff{}".format(idx)) + self.get_random(),
                        )
                    except:
                        pass
        return new_curves

    def fitness_scale_grading(self, pop, scaling=True):
        """Calculate fitness, scale, and grade for a population.

        :param list() pop: Population of individual, i.e. list of curves
        :param boolean scaling: Linearly scale fitness scores
        :return: List sets of curves graded by fitness scores
        :rtype: list()

        """
        # Intial fitness calcs
        fitnesses = [self.determine_fitness(curves) for curves in pop]

        # Scaling
        pop_scaled = self.scale_fitnesses(fitnesses, pop, scaling)

        # Grading
        pop_graded = self.grade_population(pop_scaled)

        return pop_graded

    def evolve_population(self, pop):
        """Evolve population to create a new generation.

        :param list() pop: Population of individual, i.e. list of curves

        """
        # Fitness, Scaling, Grading
        pop_graded = self.fitness_scale_grading(pop)

        # Retain best performers as parents
        retain_length = int(len(pop_graded) * self.retain)
        parents = pop_graded[:retain_length]

        # Randomly add other individuals to
        # promote genetic diversity
        for individual in pop_graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Mutate some individuals
        for idx, individual in enumerate(parents):
            if self.mutate > random.random():
                parents[idx] = self.perform_mutation(individual)

        # Crossover parents to create children
        self.perform_crossover(parents)
        self.identify_best_performer()

    def determine_fitness(self, set_of_curves):
        """Compute fitness score of an individual.

        :param SetofCurves() set_of_curves: Set of curves
        :return: Fitness score
        :rtype: float

        """
        # Temporary assign curve to equipment
        self.equipment.set_of_curves = set_of_curves.curves

        # Compute normalization score
        # and RSME with base curve
        # TODO: Try PCM
        # TODO: Try Frechet distance
        # TODO: Area between two curves
        # TODO: Dynamic Time Warping distance
        curve_normal_score = 0
        rsme = 0
        for c in set_of_curves.curves:
            if c.out_var in self.vars:
                curve_normal_score += abs(1 - c.get_out_reference())
                x, y = set_of_curves.get_data_for_plotting(c, False)
                base_x, base_y = self.base_curves_data[c.out_var]
                rsme += np.sqrt(((np.array(y) - np.array(base_y)) ** 2).mean())

        if self.equipment.type == "chiller":
            iplv_score = abs(self.equipment.calc_eff(eff_type="iplv") - self.target)
            full_eff_score = abs(
                self.equipment.calc_eff(eff_type="kwpton") - self.equipment.full_eff
            )
            iplv_weight = 1
            eff_weight = 1
            curve_normal_score_weight = 1
            rsme_weight = 1 #0.5
            fit_score = (
                iplv_score * iplv_weight
                + full_eff_score * eff_weight
                + curve_normal_score * curve_normal_score_weight
                + rsme * rsme_weight
            ) / (
                iplv_weight
                + eff_weight
                + curve_normal_score_weight * len(set_of_curves.curves)
                + rsme_weight
            )
        else:
            raise ValueError("This type of equipment has not yet been implemented.")

        return fit_score

    def scale_fitnesses(self, fitnesses, pop, scaling=True):
        """Scale the fitness scores to prevent best performers from dragging the whole population to a local extremum.

        :param list() fitnesses: List of fitness for each set of curves
        :param list() pop: List of sets of curves
        :param boolean scaling: Specifies whether of not to linearly scale the fitnesses
        :return: List of tuples representing the fitness of a set of curves and the set of curves
        :rtype: list(tuple())

        """
        # linear scaling: a + b * f
        # Based on "Scaling in Genetic Algorithms", Sushil J. Louis
        # https://www.cse.unr.edu/~sushil/class/gas/notes/scaling/index.html
        if scaling:
            max_f = max(fitnesses)
            min_f = min(fitnesses)
            avg_f = sum(fitnesses) / len(fitnesses)
            if min_f > (self.sFac * avg_f - max_f) / (self.sFac - 1.0):
                d = max_f - avg_f
                if d == 0:
                    a = 1
                    b = 0
                else:
                    a = (self.sFac - 1.0) * avg_f
                    b = avg_f * (max_f - (self.sFac * avg_f)) / d
            else:
                d = avg_f - min_f
                if d == 0:
                    a = 1
                    b = 0
                else:
                    a = avg_f / d
                    b = -min_f * avg_f
        else:
            a = 1
            b = 0

        pop_scaled = [
            (a * self.determine_fitness(set_of_curves) + b, set_of_curves)
            for set_of_curves in pop
        ]
        return pop_scaled

    def grade_population(self, pop_scaled):
        """Grade population.

        :param list(tuple()) pop_scaled: List of tuples representing the fitness of a set of curves and the set of curves
        :return: List of set of curves graded from the best to the worst
        :rtype: list(SetofCurves())

        """
        pop_sorted = sorted(pop_scaled, key=lambda tup: tup[0])
        pop_graded = [item[1] for item in pop_sorted]
        return pop_graded

    def perform_mutation(self, individual):
        """Mutate individual.

        :param SetofCurves() individual: Set of curves
        :return: Modified indivudal
        :rtype: SetofCurves()

        """
        new_individual = copy.deepcopy(individual)
        for curve in new_individual.curves:
            if len(self.vars) == 0 or curve.out_var in self.vars:
                idx = random.randint(1, curve.nb_coeffs())
                setattr(
                    curve,
                    "coeff{}".format(idx),
                    getattr(curve, "coeff{}".format(idx)) + self.get_random(),
                )
        return new_individual

    def perform_crossover(self, parents):
        """Crossover best individuals.

        :param list() parents: List of best performing individuals of the generation

        """
        parents_length = len(parents)
        desired_length = len(self.pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)
            # Can't crossover with the same element
            if male != female:
                male = parents[male]
                female = parents[female]
                child = SetofCurves(eqp_type=self.equipment.type)
                curves = []
                # male and female curves are structured the same way
                for _, c in enumerate(male.curves):
                    # Copy as male
                    n_child_curves = copy.deepcopy(c)
                    if c.out_var in self.vars or len(self.vars) == 0:
                        if c.type == "quad":
                            positions = [[1], [2, 3]]  # cst  # x^?
                        elif c.type == "bi_quad":
                            positions = [
                                [1],  # cst
                                [2, 3],  # x^?
                                [4, 5],  # y^?
                                [6],
                            ]  # x*y
                        elif c.type == "bi_cub":
                            positions = [
                                [1],  # cst
                                [2, 3, 7],  # x^?
                                [4, 5, 8],  # y^?
                                [6],  # x*y
                                [9],  # x^2*y
                                [10],
                            ]  # x*y^2
                        else:
                            raise ValueError("Type of curve not yet implemented.")
                        couple = ["male", copy.deepcopy(female)]
                        cnt = 0
                        for p in positions:
                            # Alternate between male and female
                            cnt = (cnt + 1) % 2
                            if couple[cnt] != "male":
                                # sub_position
                                for s_p in p:
                                    setattr(
                                        n_child_curves,
                                        "coeff{}".format(s_p),
                                        getattr(n_child_curves, "coeff{}".format(s_p)),
                                    )
                    curves.append(n_child_curves)
                child.curves = curves
                children.append(child)
        parents.extend(children)
        self.pop = parents

    def identify_best_performer(self):
        """Assign the best individual to the equipment.

        """
        # Re-compute fitness, scaling and grading
        pop_graded = self.fitness_scale_grading(self.pop, scaling=True)

        # Assign the equipment with the best fitness
        self.equipment.set_of_curves = pop_graded[0].curves
