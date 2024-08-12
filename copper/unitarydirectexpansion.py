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
        ref_cap_unit="si",
        fan_power=None,
        part_eff=0,
        ref_gross_cap=None,
        ref_net_cap=None,
        part_eff_unit="",
        set_of_curves="",
        part_eff_ref_std="ahri_340/360",
        part_eff_ref_std_alt=None,
        model="simplified_bf",
        sim_engine="energyplus",
        condenser_type="air",
        fan_control_mode="constant_speed",
    ):
        global log_fan
        self.type = "UnitaryDirectExpansion"
        if model != "simplified_bf":
            logging.error("Model must be 'simplified_bf'")
            raise ValueError("Model must be 'simplified_bf'")
        if ref_gross_cap == None:
            if ref_net_cap == None:
                logging.error("Input must be one and only one capacity input")
                raise ValueError("Input must be one and only one capacity input")
            else:
                if fan_power == None:
                    fan_power = 0.28434517 * ref_net_cap * 400 * 0.365 / 1000
                    # This is 400 cfm/ton and 0.365 W/cfm. Equation 11.1 from AHRI 210/240.
                    if not log_fan:
                        logging.info(f"Default fan power used: {fan_power} kW")
                        log_fan = True
                ref_gross_cap = ref_net_cap + fan_power
        else:
            if ref_net_cap != None:
                logging.error("Input must be one and only one capacity input")
                raise ValueError("Input must be one and only one capacity input")
            if fan_power == None:
                fan_power = (
                    0.28434517
                    * 400
                    * 0.365
                    * (ref_gross_cap / 1000)
                    / (1 + 0.28434517 * 400 * 0.365)
                )
                if not log_fan:
                    logging.info(f"Default fan power used: {fan_power} kW")
                    log_fan = True
            ref_net_cap = ref_gross_cap - fan_power
        self.ref_cap_unit = ref_cap_unit
        if self.ref_cap_unit != "si":
            ref_net_cap_ton = Units(value=ref_net_cap, unit=self.ref_cap_unit)
            self.ref_net_cap = ref_net_cap_ton.conversion(new_unit="kW")
            ref_gross_cap_ton = Units(value=ref_gross_cap, unit=self.ref_cap_unit)
            self.ref_gross_cap = ref_gross_cap_ton.conversion(new_unit="kW")
        else:
            self.ref_net_cap = ref_net_cap
            self.ref_gross_cap = ref_gross_cap
        self.fan_power = fan_power
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.compressor_type = compressor_type
        self.set_of_curves = set_of_curves
        self.part_eff_ref_std = part_eff_ref_std
        self.model = model
        self.sim_engine = sim_engine
        self.part_eff_ref_std_alt = part_eff_ref_std_alt
        self.condenser_type = condenser_type
        self.fan_control_mode = fan_control_mode
        self.compressor_speed = compressor_speed
        self.ref_cap_unit = ref_cap_unit
        # Define rated temperatures
        # air entering drybulb,air entering wetbulb, outdoor enter, outdoor leaving
        aed, aew, ect, lct = self.get_rated_temperatures()
        ect = ect[0]
        # Defined plotting ranges and (rated) temperature for normalization
        nb_val = 50
        if self.model == "simplified_bf":
            self.plotting_range = {
                "eir-f-t": {
                    "x1_min": aew,
                    "x1_max": aew,
                    "x1_norm": aew,
                    "nbval": nb_val,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "cap-f-t": {
                    "x1_min": aew,
                    "x1_max": aew,
                    "x1_norm": aew,
                    "nbval": 50,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "eir-f-f": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
                "cap-f-f": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
                "plf-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
            }

    def calc_rated_eff(
        self, eff_type="ieer", unit="eer", output_report=False, alt=False
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
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std
        # Retrieve curves
        curves = self.get_dx_curves()
        cap_f_f = curves["cap_f_ff"]
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_f = curves["eir_f_ff"]
        plf_f_plr = curves["plf_f_plr"]
        tot_cap_flow_mod_fac = cap_f_f.evaluate(1, 1)
        eir_flow_mod_fac = eir_f_f.evaluate(1, 1)
        num_of_reduced_cap = equipment_references[self.type][std]["coef"][
            "numofreducedcap"
        ]
        reduced_plr = equipment_references[self.type][std]["coef"]["reducedplr"]
        weighting_factor = equipment_references[self.type][std]["coef"][
            "weightingfactor"
        ]
        tot_cap_temp_mod_fac = cap_f_t.evaluate(
            equipment_references[self.type][std][
                "cooling_coil_inlet_air_wet_bulb_rated"
            ],
            equipment_references[self.type][std][
                "outdoor_unit_inlet_air_dry_bulb_rated"
            ],
        )
        net_cooling_cap_rated = (
            self.ref_gross_cap * tot_cap_temp_mod_fac * tot_cap_flow_mod_fac
            - self.fan_power
        )
        rated_cop = self.full_eff
        ieer = 0
        for red_cap_num in range(num_of_reduced_cap):
            if reduced_plr[red_cap_num] > 0.444:
                outdoor_unit_inlet_air_dry_bulb_temp_reduced = (
                    5.0 + 30.0 * reduced_plr[red_cap_num]
                )
            else:
                outdoor_unit_inlet_air_dry_bulb_temp_reduced = equipment_references[
                    self.type
                ][std]["outdoor_unit_inlet_air_dry_bulb_reduced"]
            tot_cap_temp_mod_fac = cap_f_t.evaluate(
                equipment_references[self.type][std][
                    "cooling_coil_inlet_air_wet_bulb_rated"
                ],
                outdoor_unit_inlet_air_dry_bulb_temp_reduced,
            )
            net_cooling_cap_reduced = (
                self.ref_gross_cap * tot_cap_temp_mod_fac * tot_cap_flow_mod_fac
                - self.fan_power
            )
            eir_temp_mod_fac = eir_f_t.evaluate(
                equipment_references[self.type][std][
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
            load_factor = (
                reduced_plr[red_cap_num]
                * net_cooling_cap_rated
                / net_cooling_cap_reduced
                if net_cooling_cap_reduced > 0.0
                else 1.0
            )
            degradation_coeff = 1.130 - 0.130 * load_factor
            elec_power_reduced_cap = (
                degradation_coeff
                * eir
                * (self.ref_gross_cap * tot_cap_temp_mod_fac * tot_cap_flow_mod_fac)
            )
            eer_reduced = (load_factor * net_cooling_cap_reduced) / (
                load_factor * elec_power_reduced_cap + self.fan_power
            )
            ieer += weighting_factor[red_cap_num] * eer_reduced
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

        ref_net_cap = self.ref_net_cap

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
                curves["cap_f_t"] = curve
            elif curve.out_var == "cap-f-ff":
                curves["cap_f_ff"] = curve
            elif curve.out_var == "eir-f-t":
                curves["eir_f_t"] = curve
            elif curve.out_var == "eir-f-ff":
                curves["eir_f_ff"] = curve
            elif curve.out_var == "plf-f-plr":
                curves["plf_f_plr"] = curve
        return curves

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
        dx_data = equipment_references[self.type][std][self.condenser_type]
        # Air Entering Indoor Drybulb
        aed = [
            Equipment.convert_to_deg_c(t, dx_data["ae_unit"]) for t in dx_data["aed"]
        ]
        # Air Entering Indoor Wetbulb
        aew = [
            Equipment.convert_to_deg_c(t, dx_data["ae_unit"]) for t in dx_data["aew"]
        ]
        # Outdoor Water/Air entering
        ect = [
            Equipment.convert_to_deg_c(t, dx_data["ect_unit"]) for t in dx_data["ect"]
        ]
        # Outdoor Water/Air leaving
        lct = Equipment.convert_to_deg_c(dx_data["lct"], dx_data["lct_unit"])
        return [aed, aew, ect, lct]

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
        norm_val = {"ect_lwt": self.ref_ect, "lct_lwt": self.ref_lct}[self.model]

        ranges = {
            "eir-f-t": {
                "vars_range": [(4, 10), (10.0, 40.0)],
                "normalization": (self.ref_lwt, norm_val),
            },
            "eir-f-f": {"vars_range": [(0.0, 1.0)], "normalization": (1.0)},
            "cap-f-t": {
                "vars_range": [(4, 10), (10.0, 40.0)],
                "normalization": (self.ref_lwt, norm_val),
            },
            "cap-f-f": {"vars_range": [(0.0, 1.0)], "normalization": (1.0)},
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
        assert self.compressor_type in [
            "centrifugal",
            "any",
            "positive_displacement",
            "reciprocating",
            "scroll",
            "screw",
            "scroll/screw",
        ]
        assert self.compressor_speed in ["constant", "variable", "any"]

        if lib is None or filters is None or csets is None:
            lib, filters = self.get_lib_and_filters()
            csets = self.get_curves_from_lib(lib=lib, filters=filters)

        full_eff = Units(self.full_eff, self.full_eff_unit)
        full_eff_cop = full_eff.conversion("cop")
        part_eff = Units(self.part_eff, self.part_eff_unit)
        part_eff_cop = part_eff.conversion("cop")

        self.misc_attr = {
            "model": self.model,
            "ref_net_cap": self.net_cap,
            "ref_gross_cap": self.ref_gross_cap,
            "full_eff": full_eff_cop,
            "part_eff": part_eff_cop,
            "ref_eff_unit": "",
            "compressor_type": self.compressor_type,
            "condenser_type": self.condenser_type,
            "compressor_speed": self.compressor_speed,
            "sim_engine": self.sim_engine,
            "min_plr": self.min_plr,
            "min_unloading": self.min_unloading,
            "max_plr": 1,
            "name": "Aggregated set of curves",
            "source": "Copper",
        }

        self.ranges = self.get_ranges()
        curves = SetsofCurves(sets=csets, eqp=self)
        return curves
