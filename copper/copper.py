import numpy as np
import matplotlib.pyplot as plt
import json, copy


class Library:
    def __init__(self, path="./fixtures/chiller_curves.json"):
        self.path = path
        self.data = json.loads(open(self.path, "r").read())

    def content(self):
        return self.data

    def get_unique_eqp_fields(self):
        key_val = {}
        for eqp_n, eqp_f in self.data.items():
            for field, val in eqp_f.items():
                if field != "curves" and field != "name":
                    if field not in key_val.keys():
                        key_val[field] = [val]
                    else:
                        key_val[field].append(val)
        for key, val in key_val.items():
            key_val[key] = set(val)
        return key_val

    def find_curve_sets_from_lib(self, filters=[]):
        """
        Retrieve curve sets from a JSON library that match the specified filters (tuples of strings)
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
        """
        Find equipment matching specified filter in the curve library
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

    def get_curve_set_by_name(self, name, eqp_match="chiller"):
        """
        Retrieve curve set from the library by name
        """
        # Initialize curve set object
        c_set = CurveSet(eqp_match)
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
        # Initialize curve object
        c_obj = Curve(eqp_type)
        c_obj.out_var = c
        # Curve properties
        c_prop = c_name["curves"][c]
        # Retrive all attributes of the curve object
        for c_att in list(Curve(eqp_type).__dict__):
            # Set the attribute of new Curve object
            # if attrubute are identified in database entry
            if c_att in list(c_prop.keys()):
                c_obj.__dict__[c_att] = c_prop[c_att]
        return c_obj


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
        model="ect&lwt",
        sim_engine="energyplus",
        full_rating="kwpton",
        part_rating="iplv",
    ):
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

    def find_base_curves(self):
        """
        Find an existing equipment curve that best matches the chiller
        """

        lib = Library(path="./fixtures/chiller_curves.json")

        # Define chiller properties
        props = [
            ("eqp_type", "chiller"),
            ("compressor_type", self.compressor_type),
            ("condenser_type", self.condenser_type),
            ("compressor_speed", self.compressor_speed),
            ("sim_engine", self.sim_engine),
            ("algorithm", self.model),
        ]

        # Find equipment match in the library
        eqp_match = lib.find_equipment(filters=props)

        if len(eqp_match) > 0:
            # If multiple equipment match the specified properties,
            # return the one that has numeric attributes that best
            # match the proposed case
            if len(eqp_match) > 1:
                return lib.get_curve_set_by_name(ga.get_best_match(self, eqp_match))
            else:
                return lib.get_curve_set_by_name(eqp_match)
        else:
            raise ValueError(
                "Could not find a set of curves that matches the specified properties."
            )

    def iplv(self, unit="kWpton"):
        """
        Calculate chiller IPLV
        """

        # Retrieve equipment efficiency and unit
        kwpton_ref = self.full_eff
        kwpton_ref_unit = self.full_eff_unit

        # Convert to kWpton if necessary
        if self.full_eff_unit != "kWpton":
            kwpton_ref_unit = Unit(kwpton_ref, kwpton_ref_unit)
            kwpton_ref = kwpton_ref_unit.conversion("kWpton")

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
        if self.model == "ect&lwt":
            if self.condenser_type == "air":
                # Temperatures from AHRI Std 550/590
                chw = 6.67
                ect = [3 + 32 * loads[0], 3 + 32 * loads[1], 3 + 32 * loads[2], 13]
            elif self.condenser_type == "water":
                # Temperatures from AHRI Std 550/590
                chw = 6.67
                ect = [8 + 22 * loads[0], 8 + 22 * loads[1], 19, 19]

            # Retrieve curves
            for curve in self.curveset.curves:
                if curve.output_var == "cap-f-T":
                    cap_f_t = curve
                elif curve.output_var == "eir-f-T":
                    eir_f_t = curve
                else:
                    eir_f_plr = curve

            # Calculate EIR for each testing conditions
            for idx, load in enumerate(loads):
                dt = ect[idx] - chw
                cap_f_chw_ect = cap_f_t.evaluate(chw, ect[idx])
                eir_f_chw_ect = eir_f_t.evaluate(chw, ect[idx])
                cap_op = load_ref * cap_f_chw_ect
                plr = load / cap_op
                eir_plr = eir_f_plr.evaluate(plr, dt)
                # eir = power / load so eir * plr = (power / load) * (load / cap_op)
                eir = eir_ref * eir_f_chw_ect * eir_plr / plr
                kwpton_lst.append(eir / kbtu_to_kw * ton_to_kbtu)

            # Coefficients from AHRI Std 550/590
            iplv = 1 / (
                (0.01 / kwpton_lst[0])
                + (0.42 / kwpton_lst[1])
                + (0.45 / kwpton_lst[2])
                + (0.12 / kwpton_lst[3])
            )
        else:
            # TODO:
            # Implement IPLV calcs for other chiller algorithm
            raise ValueError("Algorithm not implemented.")

        # Convert IPLV to desired unit
        if unit != "kWpton":
            iplv_org = Unit(iplv, "kWpton")
            iplv = iplv_org.conversion(unit)

        return iplv

    def kwpton(self, value):
        pass


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
                                    self.plotting_range[var]["x1_norm"], self.plotting_range[var]["x2_norm"]
                                )
                                if norm
                                else 1
                            )
                            y_val = curve.evaluate(x_1_vals[v], x_2_vals[v]) / norm_fac
                        else:
                            norm_fac = (
                                curve.evaluate(
                                    self.plotting_range[var]["x1_norm"], self.plotting_range[var]["x1_norm"]
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
    def __init__(self, eqp_type):
        # General charactersitics
        self.out_var = ""
        self.type = ""
        self.units = "si"
        self.type = ""
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.ref_x = 0
        self.ref_y = 0
        self.out_min = 0
        self.out_max = 0
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


class Unit:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_unit):
        """
        Convert efficiency rating
        """
        if new_unit == "kWpton":
            if self.unit == "COP":
                return 12.0 / (self.value * 3.412)
            elif self.unit == "kWpton":
                return self.value
            elif self.unit == "EER":
                return 12.0 / self.value
            else:
                return self.value
        elif new_unit == "COP":
            if self.unit == "kWpton":
                return 12.0 / self.value / 3.412
            elif self.unit == "COP":
                return self.value
            elif self.unit == "EER":
                return self.value / 3.412
            else:
                return self.value
        elif new_unit == "EER":
            if self.unit == "kWpton":
                return 12.0 / self.value
            elif self.unit == "EER":
                return self.value
            elif self.unit == "COP":
                return 3.412 * self.value
            else:
                return self.value
        elif new_unit == "ton":
            if self.unit == "kW":
                return self.value * (3412 / 12000)
        elif new_unit == "kW":
            if self.unit == "ton":
                return self.value / (3412 / 12000)
