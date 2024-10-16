"""
unitary_dx.py
====================================
This is the unitary direct expansion (DX) HVAC equipment module of Copper. The module handles all calculations and data manipulation related to unitary DX equipment.
"""
import CoolProp.CoolProp as CP
from copper.generator import *
from copper.units import *
from copper.curves import *
from copper.library import *
from copper.equipment import *
import logging, json

location = os.path.dirname(os.path.realpath(__file__))
unitary_dx_lib = os.path.join(location, "data", "unitarydirectexpansion_curves.json")
equipment_references = json.load(
    open(os.path.join(location, "data", "equipment_references.json"), "r")
)
log_fan = False


class UnitaryDirectExpansion(Equipment):
    def __init__(
        self,
        full_eff,
        full_eff_unit,
        compressor_type,
        compressor_speed="constant",
        ref_cap_unit="W",
        indoor_fan_power=None,
        part_eff=0,
        ref_gross_cap=None,
        ref_net_cap=None,
        part_eff_unit="",
        set_of_curves=[],
        part_eff_ref_std="ahri_340/360",
        part_eff_ref_std_alt=None,
        model="simplified_bf",
        sim_engine="energyplus",
        condenser_type="air",
        degradation_coefficient=0.115170535550221,  # PLF = 1 - (1 - PLR) * C_D = (1 - C_D) + C_D * PLR Equation 11.63 in AHRI 240/210 (2024)
        indoor_fan_speeds_mapping={
            "1": {
                "fan_flow_fraction": 0.66,
                "fan_power_fraction": 0.4,
                "capacity_fraction": 0.5,
            },
            "2": {
                "fan_flow_fraction": 1.0,
                "fan_power_fraction": 1.0,
                "capacity_fraction": 1.0,
            },
        },
        indoor_fan_curve_coef={
            "type": "cubic",
            "1": 0.63 * 0.0408,
            "2": 0.63 * 0.088,
            "3": -0.63 * 0.0729,
            "4": 0.63 * 0.9437,
        },
        indoor_fan_speeds=1,
        indoor_fan_curve=False,
        indoor_fan_power_unit="kW",
    ):
        global log_fan
        self.type = "UnitaryDirectExpansion"

        # Input validation and populate default assumptions
        if model != "simplified_bf":
            logging.error("Model must be 'simplified_bf'")
            raise ValueError("Model must be 'simplified_bf'")
        if ref_gross_cap == None:
            if ref_net_cap == None:
                logging.error("Input must be one and only one capacity input")
                raise ValueError("Input must be one and only one capacity input")
            else:
                if indoor_fan_power == None:
                    # This is 400 cfm/ton and 0.365 W/cfm. Equation 11.1 from AHRI 210/240 (2024).
                    indoor_fan_power_unit = "kW"
                    indoor_fan_power_unit = "kW"
                    indoor_fan_power = Units(
                        value=Units(value=ref_net_cap, unit=ref_cap_unit).conversion(
                            new_unit="ton"
                        )
                        * 400
                        * 0.365,
                        unit="W",
                    ).conversion(new_unit=indoor_fan_power_unit)
                    if not log_fan:
                        logging.info(
                            f"Default fan power is based on 400 cfm/ton and 0.365 kW/cfm"
                        )
                        logging.info(
                            f"Default fan power is based on 400 cfm/ton and 0.365 kW/cfm"
                        )
                        log_fan = True
                ref_gross_cap = Units(
                    value=Units(value=ref_net_cap, unit=ref_cap_unit).conversion(
                        new_unit=indoor_fan_power_unit
                    )
                    + Units(
                        value=indoor_fan_power, unit=indoor_fan_power_unit
                    ).conversion(new_unit="kW"),
                    unit="kW",
                ).conversion(ref_cap_unit)
        else:
            if ref_net_cap != None:
                logging.error("Input must be one and only one capacity input")
                raise ValueError("Input must be one and only one capacity input")
            if indoor_fan_power == None:
                # This is 400 cfm/ton and 0.365 W/cfm. Equation 11.1 from AHRI 210/240 (2024).
                indoor_fan_power_unit = "kW"
                indoor_fan_power = Units(
                    value=(
                        400
                        * 0.365
                        * Units(value=ref_gross_cap, unit=ref_cap_unit).conversion(
                            new_unit="ton"
                        )
                    )
                    / (
                        1
                        + 400
                        * 0.365
                        * Units(value=1.0, unit=ref_cap_unit).conversion(new_unit="ton")
                        * Units(value=1.0, unit="W").conversion(new_unit=ref_cap_unit)
                    ),
                    unit="W",
                ).conversion(new_unit=indoor_fan_power_unit)
                if not log_fan:
                    logging.info(f"Default fan power used: {indoor_fan_power} kW")
                    log_fan = True
            ref_net_cap = Units(
                value=Units(value=ref_gross_cap, unit=ref_cap_unit).conversion(
                    new_unit=indoor_fan_power_unit
                )
                - indoor_fan_power,
                unit=indoor_fan_power_unit,
            ).conversion(ref_cap_unit)
        self.ref_cap_unit = ref_cap_unit
        if self.ref_cap_unit != "kW":
            ref_net_cap_ton = Units(value=ref_net_cap, unit=self.ref_cap_unit)
            self.ref_net_cap = ref_net_cap_ton.conversion(new_unit="kW")
            ref_gross_cap_ton = Units(value=ref_gross_cap, unit=self.ref_cap_unit)
            self.ref_gross_cap = ref_gross_cap_ton.conversion(new_unit="kW")
            self.ref_cap_unit = "kW"
        else:
            self.ref_net_cap = ref_net_cap
            self.ref_gross_cap = ref_gross_cap
            self.ref_cap_unit = ref_cap_unit

        # Get attributes
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.full_eff_alt = 0
        self.full_eff_alt_unit = full_eff_unit
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.part_eff_alt = 0
        self.part_eff_alt_unit = part_eff_unit
        self.compressor_type = compressor_type
        self.set_of_curves = set_of_curves
        self.part_eff_ref_std = part_eff_ref_std
        self.model = model
        self.sim_engine = sim_engine
        self.part_eff_ref_std_alt = part_eff_ref_std_alt
        self.condenser_type = condenser_type
        self.compressor_speed = compressor_speed
        self.indoor_fan_speeds_mapping = indoor_fan_speeds_mapping
        self.indoor_fan_speeds = indoor_fan_speeds
        self.indoor_fan_power = indoor_fan_power
        self.indoor_fan_curve_coef = indoor_fan_curve_coef
        self.indoor_fan_power_unit = indoor_fan_power_unit
        self.indoor_fan_curve = indoor_fan_curve
        # Define rated temperatures
        # air entering drybulb, air entering wetbulb, entering condenser temperature, leaving condenser temperature
        aed, self.aew, ect, lct = self.get_rated_temperatures()
        self.ect = ect[0]

        # Defined plotting ranges and (rated) temperature for normalization
        nb_val = 50
        if self.model == "simplified_bf":
            self.plotting_range = {
                "eir-f-t": {
                    "x1_min": self.aew,
                    "x1_max": self.aew,
                    "x1_norm": self.aew,
                    "nbval": nb_val,
                    "x2_min": 15,
                    "x2_max": 40,
                    "x2_norm": self.ect,
                },
                "cap-f-t": {
                    "x1_min": self.aew,
                    "x1_max": self.aew,
                    "x1_norm": self.aew,
                    "nbval": 50,
                    "x2_min": 15,
                    "x2_max": 40,
                    "x2_norm": self.ect,
                    "nbval": nb_val,
                },
                "eir-f-ff": {"x1_min": 0, "x1_max": 2, "x1_norm": 1, "nbval": nb_val},
                "cap-f-ff": {"x1_min": 0, "x1_max": 2, "x1_norm": 1, "nbval": nb_val},
                "plf-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
            }

        # Cycling degradation
        self.degradation_coefficient = degradation_coefficient
        self.add_cycling_degradation_curve()

    def add_cycling_degradation_curve(self, overwrite=False, return_curve=False):
        """Determine and assign a part load fraction as a function of part load ratio curve to a unitary DX equipment.

        :param str overwrite: Flag to overwrite the existing degradation curve. Default is False.
        :param str assign_curve: Add curve to equipment's. Default is True.
        """
        # Remove exisiting curve if it exists
        if overwrite:
            for curve in self.set_of_curves:
                if curve.out_var == "plf-f-plr":
                    self.set_of_curves.remove(curve)
                    break

        # Add new curve
        if not "plf-f-plr" in self.get_dx_curves().keys() or overwrite:
            plf_f_plr = Curve(eqp=self, c_type="linear")
            plf_f_plr.out_var = "plf-f-plr"
            plf_f_plr.type = "linear"
            plf_f_plr.coeff1 = 1 - self.degradation_coefficient
            plf_f_plr.coeff2 = self.degradation_coefficient
            plf_f_plr.x_min = 0
            plf_f_plr.x_max = 1
            plf_f_plr.out_min = 0
            plf_f_plr.out_max = 1
            if return_curve:
                return plf_f_plr
            else:
                self.set_of_curves.append(plf_f_plr)

        # default fan curve
        self.default_fan_curve = Curve(
            eqp=self, c_type=self.indoor_fan_curve_coef["type"]
        )
        self.default_fan_curve.coeff1 = self.indoor_fan_curve_coef["1"]
        self.default_fan_curve.coeff2 = self.indoor_fan_curve_coef["2"]
        self.default_fan_curve.coeff3 = self.indoor_fan_curve_coef["3"]
        self.default_fan_curve.coeff4 = self.indoor_fan_curve_coef["4"]

    def calc_fan_power(self, capacity_fraction):
        """Calculate unitary DX equipment fan power.

        :param float capacity_fraction: Ratio of actual capacity to net rated capacity
        :return: Unitary DX Equipment fan power in Watts
        :rtype: float

        """
        # Full flow/power
        flow_fraction = capacity_fraction  # we assume flow_fraction = 1*capacity_fraction as default
        if capacity_fraction == 1 or self.indoor_fan_speeds == 1:
            return self.indoor_fan_power
        else:
            if self.indoor_fan_curve == False:
                capacity_fractions = []
                fan_power_fractions = []
                for speed_info in self.indoor_fan_speeds_mapping.values():
                    capacity_fractions.append(speed_info["capacity_fraction"])
                    fan_power_fractions.append(speed_info["fan_power_fraction"])
                # Minimum flow/power
                if capacity_fraction <= capacity_fractions[0]:
                    return self.indoor_fan_power * fan_power_fractions[0]
                elif capacity_fraction in capacity_fractions:
                    return (
                        self.indoor_fan_power
                        * fan_power_fractions[
                            capacity_fractions.index(capacity_fraction)
                        ]
                    )
                else:
                    # In between-speeds: determine power by linear interpolation
                    for i, ratio in enumerate(capacity_fractions):
                        if (
                            ratio < capacity_fraction
                            and capacity_fractions[i + 1] > capacity_fraction
                        ):
                            a = (
                                fan_power_fractions[i + 1] - fan_power_fractions[i]
                            ) / (capacity_fractions[i + 1] - capacity_fractions[i])
                            b = fan_power_fractions[i] - a * capacity_fractions[i]
                            return self.indoor_fan_power * (a * capacity_fraction + b)
            else:  # using curve
                default_min_fan_power = (
                    self.indoor_fan_power * 0.25
                )  # default min fan power
                power_factor = self.default_fan_curve.evaluate(x=flow_fraction, y=0)
                if self.indoor_fan_power * power_factor > default_min_fan_power:
                    return self.indoor_fan_power * power_factor
                else:
                    return default_min_fan_power

    def calc_rated_eff(
        self, eff_type="part", unit="cop", output_report=False, alt=False
    ):
        """Calculate unitary DX equipment efficiency.

        :param str eff_type: Unitary DX equipment efficiency type, currently supported `full` (full load rating)
                            and `part` (part load rating)
        :param str unit: Efficiency unit
        :param bool output_report: Indicate output report generation
        :param bool alt: Indicate the DX system alternate standard rating should be used
        :return: Unitary DX Equipment rated efficiency
        :rtype: float

        """

        # Handle alternate ratings (not currently used)
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std

        # Retrieve curves
        curves = self.get_dx_curves()
        cap_f_f = curves["cap-f-ff"]
        cap_f_t = curves["cap-f-t"]
        eir_f_t = curves["eir-f-t"]
        eir_f_f = curves["eir-f-ff"]
        if not "plf-f-plr" in curves.keys():
            self.add_cycling_degradation_curve()
            curves = self.get_dx_curves()
        plf_f_plr = curves["plf-f-plr"]

        # Calculate capacity and efficiency degradation as a function of flow fraction
        tot_cap_flow_mod_fac = cap_f_f.evaluate(1, 1)
        eir_flow_mod_fac = eir_f_f.evaluate(1, 1)

        # Get rated conditions
        eqp_type = self.type.lower()
        num_of_reduced_cap = equipment_references[eqp_type][std]["coef"][
            "numofreducedcap"
        ]
        reduced_plr = equipment_references[eqp_type][std]["coef"]["reducedplr"]
        weighting_factor = equipment_references[eqp_type][std]["coef"][
            "weightingfactor"
        ]
        tot_cap_temp_mod_fac = cap_f_t.evaluate(
            equipment_references[eqp_type][std][
                "cooling_coil_inlet_air_wet_bulb_rated"
            ],
            equipment_references[eqp_type][std][
                "outdoor_unit_inlet_air_dry_bulb_rated"
            ],
        )

        # Calculate NET rated capacity
        net_cooling_cap_rated = (
            self.ref_gross_cap * tot_cap_temp_mod_fac * tot_cap_flow_mod_fac
            - self.indoor_fan_power
        )

        # Convert user-specified full load efficiency to COP
        # User-specified capacity is a NET efficiency
        full_eff = Units(value=self.full_eff, unit=self.full_eff_unit)
        rated_cop = full_eff.conversion(new_unit="cop")

        # Iterate through the different sets of rating conditions to calculate IEER
        ieer = 0
        for red_cap_num in range(num_of_reduced_cap):
            # Determine the outdoor air conditions based on AHRI Standard
            if reduced_plr[red_cap_num] > 0.444:
                outdoor_unit_inlet_air_dry_bulb_temp_reduced = (
                    5.0 + 30.0 * reduced_plr[red_cap_num]
                )
            else:
                outdoor_unit_inlet_air_dry_bulb_temp_reduced = equipment_references[
                    eqp_type
                ][std]["outdoor_unit_inlet_air_dry_bulb_reduced"]

            # Calculate capacity at rating conditions
            tot_cap_temp_mod_fac = cap_f_t.evaluate(
                equipment_references[eqp_type][std][
                    "cooling_coil_inlet_air_wet_bulb_rated"
                ],
                outdoor_unit_inlet_air_dry_bulb_temp_reduced,
            )
            load_factor_gross = min(
                1.0, (reduced_plr[red_cap_num] / tot_cap_temp_mod_fac)
            )  # Load percentage * Rated gross capacity / Available gross capacity
            indoor_fan_power = self.calc_fan_power(load_factor_gross) / 1000
            net_cooling_cap_reduced = (
                self.ref_gross_cap * tot_cap_temp_mod_fac * tot_cap_flow_mod_fac
                - indoor_fan_power
            )

            # Calculate efficency at rating conditions
            eir_temp_mod_fac = eir_f_t.evaluate(
                equipment_references[eqp_type][std][
                    "cooling_coil_inlet_air_wet_bulb_rated"
                ],
                outdoor_unit_inlet_air_dry_bulb_temp_reduced,
            )
            if rated_cop > 0.0:
                eir = eir_temp_mod_fac * eir_flow_mod_fac / rated_cop
            else:
                eir = 0.0
                logging.error("Input COP is 0!")
                raise ValueError("Input COP is 0!")

            # "Load Factor" (as per AHRI Standard) which is analogous to PLR
            if reduced_plr[red_cap_num] < 1.0:
                load_factor = (
                    reduced_plr[red_cap_num]
                    * net_cooling_cap_rated
                    / net_cooling_cap_reduced
                    if net_cooling_cap_reduced > 0.0
                    else 1.0
                )
            else:
                load_factor = 1

            # Cycling degradation
            degradation_coeff = 1 / plf_f_plr.evaluate(load_factor, 1)

            # Power
            elec_power_reduced_cap = (
                degradation_coeff
                * eir
                * (self.ref_gross_cap * tot_cap_temp_mod_fac * tot_cap_flow_mod_fac)
            )

            # EER
            eer_reduced = (load_factor * net_cooling_cap_reduced) / (
                load_factor * elec_power_reduced_cap + indoor_fan_power
            )

            if eff_type == "full":
                ieer = eer_reduced
                break

            # Update IEER
            ieer += weighting_factor[red_cap_num] * eer_reduced

        # Convert efficiency to original unit unless specified
        if unit != "cop":
            ieer = Units(value=ieer, unit="cop")
            ieer = ieer.conversion(new_unit=self.full_eff_unit)
        return ieer

    def ieer_to_eer(self, ieer):
        """Calculate EER from IEER and system capacity.
        The regression function was obtained by fitting a linear model on performance data collected from AHRI database (Sample Size = 14,268).
        Model Diagnostics:
        R-square = 0.5458
        Mean Absolute Error = 0.369
        Root Mean Square Error = 0.455
        Model was internally validated using 10-fold cross validation approach and externally validated using the USDOE database.

        :parm float ieer: Integrated energy efficiency ratio (IEER)
        :return: Energy efficiency ratio (EER)
        :rtype: float

        """

        ref_net_cap = Units(value=self.ref_net_cap, unit=self.ref_cap_unit).conversion(
            new_unit="btu/h"
        )

        eer = (
            9.886
            + 0.1804 * ieer
            - (1.88e-17) * (ref_net_cap**3)
            + (2.706e-11) * (ref_net_cap**2)
            - (1.047e-5) * (ref_net_cap)
            - (1.638e-7) * (ieer * ref_net_cap)
        )
        return eer

    def get_dx_curves(self):
        """Retrieve DX curves from the DX set_of_curves attribute.

        :return: Dictionary of the curves associated with the object
        :rtype: dict

        """
        curves = {}
        for curve in self.set_of_curves:
            if curve.out_var == "cap-f-t":
                curves["cap-f-t"] = curve
            elif curve.out_var == "cap-f-ff":
                curves["cap-f-ff"] = curve
            elif curve.out_var == "eir-f-t":
                curves["eir-f-t"] = curve
            elif curve.out_var == "eir-f-ff":
                curves["eir-f-ff"] = curve
            elif curve.out_var == "plf-f-plr":
                curves["plf-f-plr"] = curve
        return curves

    def get_curves_from_lib(self, lib, filters):
        """Function to get the sort from the library based on chiller filters.

        :param copper.library.Library lib: Chiller library object
        :param list filters: List of tuples containing the relevant filter keys and values
        :return: List of set of curves object corresponding to seed curves
        :rtype: list

        """
        sets = lib.find_set_of_curves_from_lib(filters=filters, part_eff_flag=True)
        assert sets is not None
        return sets

    def get_rated_temperatures(self, alt=False):
        """Get unitary DX equipment rated temperatures.

        :param bool alt: Indicate the unitary DX equipment alternate standard rating should be used
        :return: Rated entering condenser temperature and temperature of the air entering the system
        :rtype: list

        """
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std
        dx_data = equipment_references[self.type.lower()][std][self.condenser_type]
        # Air entering indoor dry-bulb
        aed = Equipment.convert_to_deg_c(dx_data["aed"], dx_data["ae_unit"])
        # Air entering indoor wet-bulb
        self.aew = Equipment.convert_to_deg_c(dx_data["aew"], dx_data["ae_unit"])
        # Outdoor water/air entering
        ect = [
            Equipment.convert_to_deg_c(t, dx_data["ect_unit"]) for t in dx_data["ect"]
        ]
        # Outdoor water/air leaving
        lct = Equipment.convert_to_deg_c(dx_data["lct"], dx_data["lct_unit"])
        return [aed, self.aew, ect, lct]

    def get_lib_and_filters(self, lib_path=unitary_dx_lib):
        """Get unitary DX equipment library object and unitary DX equipment specific filters.

        :param str lib_path:Full path of json library
        :return: Unitary DX equipment library object and filters
        :rtype: list

        """
        lib = Library(path=lib_path)
        filters = [
            ("eqp_type", "UnitaryDirectExpansion"),
            ("condenser_type", self.condenser_type),
            ("sim_engine", self.sim_engine),
            ("model", self.model),
        ]

        return lib, filters

    def get_ranges(self):
        """Get applicable range of values for independent variables of the unitary DX equipment model.

        :return: Range of values, and values used for normalization (reference/rated values)
        :rtype: dict

        """
        ranges = {
            "eir-f-t": {
                "vars_range": [(12.8, 26), (10.0, 40.0)],
                "normalization": (self.aew, self.ect),
            },
            "eir-f-ff": {"vars_range": [(0.0, 1.5)], "normalization": (1.0)},
            "cap-f-t": {
                "vars_range": [(12.8, 26), (10.0, 40.0)],
                "normalization": (self.aew, self.ect),
            },
            "cap-f-ff": {"vars_range": [(0.0, 1.5)], "normalization": (1.0)},
            "plf-f-plr": {"vars_range": [(0.0, 1.0)], "normalization": (1.0)},
        }

        return ranges

    def get_seed_curves(self, lib=None, filters=None, csets=None):
        """Function to generate seed curves specific to a unitary DX equipment and sets relevant attributes (misc_attr, ranges).

        :param copper.library.Library lib: Unitary DX equipment library object
        :param list filters: List of tuples containing the filter keys and values
        :param list csets: List of set of curves object corresponding to selected unitary DX equipment from library
        :rtype: SetsofCurves

        """
        assert self.compressor_type in ["scroll"]

        if lib is None or filters is None or csets is None:
            lib, filters = self.get_lib_and_filters()
            csets = self.get_curves_from_lib(lib=lib, filters=filters)

        full_eff = Units(self.full_eff, self.full_eff_unit)
        full_eff_cop = full_eff.conversion("cop")
        part_eff = Units(self.part_eff, self.part_eff_unit)
        part_eff_cop = part_eff.conversion("cop")

        self.misc_attr = {
            "model": self.model,
            "ref_net_cap": self.ref_net_cap,
            "ref_gross_cap": self.ref_gross_cap,
            "full_eff": full_eff_cop,
            "part_eff": part_eff_cop,
            "ref_eff_unit": self.full_eff_unit,
            "compressor_type": self.compressor_type,
            "condenser_type": self.condenser_type,
            "compressor_speed": self.compressor_speed,
            "sim_engine": self.sim_engine,
            "name": "Aggregated set of curves",
            "source": "Copper",
        }

        self.ranges = self.get_ranges()
        curves = SetsofCurves(sets=csets, eqp=self)
        return curves

    def get_ref_vars_for_aggregation(self):
        return ["ref_net_cap", "full_eff", "part_eff"]
