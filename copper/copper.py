"""
copper.py
====================================
This is the core module of Copper. It handles the following:

- Curves
- Curve sets
- Equipment related calculations
- Unit conversions
- Genetic algorithm
- Curve library manipulations
"""

import numpy as np
import matplotlib.pyplot as plt
import json, copy, random


class Library:
    def __init__(self, path="./fixtures/chiller_curves.json"):
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

    def find_curve_sets_from_lib(self, filters=[]):
        """Retrieve curve sets from a library matching specific filters.

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :return: All curve set object matching the filters
        :rtype: list()

        """
        # Find name of equiment that match specified filter
        eqp_match = self.find_equipment(filters)

        # List of curve sets that match specified filters
        curve_sets = []

        # Retrieve identified equipment's curve sets from the library
        for name, props in eqp_match.items():
            c_set = CurveSet(props["eqp_type"])

            # Retrive all attributes of the curve sets object
            for c_att in list(c_set.__dict__):
                # Set the attribute of new Curve object
                # if attrubute are identified in database entry
                if c_att in list(self.data[name].keys()):
                    c_set.__dict__[c_att] = self.data[name][c_att]

            c_lst = []

            # Create new CurveSet and Curve objects for all the
            # sets of curves identified as matching the filters
            for c in self.data[name]["curves"]:
                c_lst.append(
                    self.get_curve(
                        c, self.data[name], eqp_type=self.data[name]["eqp_type"]
                    )
                )
            c_set.curves = c_lst
            curve_sets.append(c_set)

        return curve_sets

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

    def get_curve_set_by_name(self, name):
        """Retrieve curve set from the library by name.

        :param str name: Curve name
        :return: Curve set object
        :rtype: CurveSet()

        """
        # Initialize curve set object
        c_set = CurveSet("chiller")
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
            # Add curves to curve set object
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

    def find_seed_curves(self, filters, eqp):
        """Find an existing equipment curve that best matches the equipment.

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :param eqp: Equipment object(e.g. Chiller())
        :return: Curve set object
        :rtype: CurveSet()

        """
        # Find equipment match in the library
        eqp_match = self.find_equipment(filters=filters)

        if len(eqp_match) > 0:
            # If multiple equipment match the specified properties,
            # return the one that has numeric attributes that best
            # match the proposed case
            if len(eqp_match) > 1:
                return self.get_curve_set_by_name(self.get_best_match(eqp, eqp_match))
            else:
                return self.get_curve_set_by_name(eqp_match)
        else:
            raise ValueError(
                "Could not find a set of curves that matches the specified properties."
            )

    def get_best_match(self, eqp, matches):
        """Find the curve set matching the equipment characteristics the best.

        :param eqp: Equipment object(e.g. Chiller())
        :param dict[str,dict[]] matches: All potential matches
        :return: Name of the curve set that best matches the equipment characteristics
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
        curveset="",
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
        self.curveset = curveset

    def generate_curve_set(
        self,
        method="typical",
        pop_size=100,
        tol=0.005,
        max_gen=15000,
        vars="",
        sFac=0.5,
        retain=0.2,
        random_select=0.3,
        mutate=0.8,
        bounds=(6, 10),
    ):
        """Generate a curve set for a particular Chiller() object.

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
        :return: Curve set object generated by the genetic algorithm that matches the Chiller() definition
        :rtype: CurveSet()

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
        )
        return ga.generate_curve_set()

    def calc_eff(self, eff_type, unit="kw/ton"):
        """
        Calculate chiller efficiency
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
            for curve in self.curveset:
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


class CurveSet:
    def __init__(self, eqp_type):
        self.name = ""
        self.curves = []
        self.validity = 1.0
        self.source = ""
        self.eqp_type = eqp_type
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

    def plot(self, out_var=[], axes=[], norm=True, color="Black", alpha=0.3):
        for i, var in enumerate(out_var):
            for curve in self.curves:
                if curve.out_var == var:
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

                    x = (
                        x_1_vals
                        if len(set(x_1_vals)) > len(set(x_2_vals))
                        else x_2_vals
                    )
                    axes[i].plot(x, y, color=color, alpha=alpha)
                    axes[i].set_title(var)
        return True


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
        self.ref_x = 0
        self.ref_y = 0
        self.out_min = 0
        self.out_max = 0
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
        """
        Return the output of a curve.
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
        """
        Find number of curve coefficients
        """
        ids = []
        for key in list(self.__dict__.keys()):
            if "coeff" in key:
                ids.append(int(key.split("coeff")[-1]))
        return max(ids)


class Unit:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_unit):
        """
        Convert efficiency rating
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
        random_select=0.05,
        mutate=0.05,
        bounds=(6, 10),
    ):
        self.equipment = equipment
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

    def generate_curve_set(self):
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

        if self.equipment.type == "chiller":
            # TODO: implement other methods
            if self.method == "typical":
                lib = Library(path="./fixtures/typical_curves.json")
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
            base_curves = lib.find_curve_sets_from_lib(filters)
        elif self.method == "best_match":
            base_curves = [lib.find_seed_curves(filters, self.equipment)]

        # Run GA
        self.run_ga(curves=base_curves)
        return self.equipment.curveset

    def run_ga(self, curves):
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
        if self.equipment.type == "chiller":
            if self.equipment.curveset != "":
                part_rating = self.equipment.calc_eff(eff_type="iplv")
                full_rating = self.equipment.calc_eff(eff_type="kwpton")
            else:
                return False
        else:
            raise ValueError("This type of equipment has not yet been implemented.")
        if (
            (part_rating < self.target * (1 + self.tol))
            and (part_rating > self.target * (1 - self.tol))
            and (full_rating < self.full_eff * (1 + self.tol))
            and (full_rating > self.full_eff * (1 - self.tol))
        ):
            return True
        else:
            return False

    def generate_population(self, curves):
        pop = []
        for _ in range(self.pop_size):
            pop.append(self.individual(curves))
        return pop

    def get_random(self):
        while True:
            val = float(
                random.randrange(-99999, 99999)
                / 10 ** (random.randint(self.bounds[0], self.bounds[1]))
            )
            if val != 0:
                return val

    def individual(self, curves):
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
        # Intial fitness calcs
        fitnesses = [self.determine_fitness(curves) for curves in pop]

        # Scaling
        pop_scaled = self.scale_fitnesses(fitnesses, pop, scaling)

        # Grading
        pop_graded = self.grade_population(pop_scaled)

        return pop_graded

    def evolve_population(self, pop):
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

    def determine_fitness(self, curveset):
        """
        Compute fitness score of an individual
        """
        # Temporary assign curve to equipment
        self.equipment.curveset = curveset.curves

        # Normalization score
        curve_normal_score = 0
        for c in curveset.curves:
            if self.equipment.type == "chiller":
                if "-t" in c.out_var:
                    if self.equipment.model == "ect_lwt":
                        x_ref = c.ref_ect
                        y_ref = c.ref_lwt
                else:
                    x_ref = 1
                    y_ref = 0
            else:
                raise ValueError("This type of equipment has not yet been implemented.")
            curve_normal_score += abs(1 - c.evaluate(x_ref, y_ref))

        if self.equipment.type == "chiller":
            iplv_score = abs(self.equipment.calc_eff(eff_type="iplv") - self.target)
            full_eff_score = abs(
                self.equipment.calc_eff(eff_type="kwpton") - self.equipment.full_eff
            )
            iplv_weight = 1
            eff_weight = 1
            fit_score = (
                iplv_score * iplv_weight
                + full_eff_score * eff_weight
                + curve_normal_score
            ) / (iplv_weight + eff_weight + len(curveset.curves))
        else:
            raise ValueError("This type of equipment has not yet been implemented.")

        return fit_score

    def scale_fitnesses(self, fitnesses, pop, scaling=True):
        """
        Scale the fitness scores to prevent best performers from draggin the whole population to a local extremum
        """
        # linear scaling: a + b * f
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
            (a * self.determine_fitness(curveset) + b, curveset) for curveset in pop
        ]
        return pop_scaled

    def grade_population(self, pop_scaled):
        pop_sorted = sorted(pop_scaled, key=lambda tup: tup[0])
        pop_graded = [item[1] for item in pop_sorted]
        return pop_graded

    def perform_mutation(self, individual):
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
                child = CurveSet(eqp_type=self.equipment.type)
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
        # Re-compute fitness, scaling and grading
        pop_graded = self.fitness_scale_grading(self.pop, scaling=True)

        # Assign the equipment with the best fitness
        self.equipment.curveset = pop_graded[0].curves
