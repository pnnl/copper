"""
chiller.py
====================================
This is the chiller module of Copper. The module handles all calculations and data manipulation related to chillers.
"""

import CoolProp.CoolProp as CP
from scipy import optimize
from copper.generator import *
from copper.units import *
from copper.curves import *
from copper.library import *
from copper.equipment import *
import logging, json

location = os.path.dirname(os.path.realpath(__file__))
chiller_lib = os.path.join(location, "data", "chiller_curves.json")
equipment_references = json.load(
    open(os.path.join(location, "data", "equipment_references.json"), "r")
)


class Chiller(Equipment):
    def __init__(
        self,
        ref_cap,
        ref_cap_unit,
        full_eff,
        full_eff_unit,
        compressor_type,
        condenser_type,
        compressor_speed,
        part_eff=0,
        part_eff_unit="",
        part_eff_ref_std="ahri_550/590",
        part_eff_alt=0,
        part_eff_unit_alt="",
        part_eff_ref_std_alt="ahri_550/590",
        full_eff_alt=0,
        full_eff_unit_alt="",
        set_of_curves="",
        model="ect_lwt",
        sim_engine="energyplus",
        min_unloading=0.1,
        min_plr=None,
    ):
        self.type = "chiller"
        self.compressor_type = compressor_type
        self.condenser_type = condenser_type
        self.compressor_speed = compressor_speed
        self.ref_cap = ref_cap
        self.ref_cap_unit = ref_cap_unit
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.full_eff_alt = full_eff_alt
        self.full_eff_unit_alt = full_eff_unit_alt
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.part_eff_ref_std = part_eff_ref_std
        self.part_eff_alt = part_eff_alt
        self.part_eff_unit_alt = part_eff_unit_alt
        self.part_eff_ref_std_alt = part_eff_ref_std_alt
        self.min_unloading = min_unloading
        self.min_plr = min_unloading if min_plr is None else min_plr
        self.model = model
        self.sim_engine = sim_engine
        self.set_of_curves = set_of_curves

        # Define rated temperatures
        ect, lwt, lct = self.get_rated_temperatures()
        ect = ect[0]

        # Defined plotting ranges and (rated) temperature for normalization
        nb_val = 50
        if self.model == "ect_lwt":
            self.plotting_range = {
                "eir-f-t": {
                    "x1_min": lwt,
                    "x1_max": lwt,
                    "x1_norm": lwt,
                    "nbval": nb_val,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "cap-f-t": {
                    "x1_min": lwt,
                    "x1_max": lwt,
                    "x1_norm": lwt,
                    "nbval": 50,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
            }
        elif self.model == "lct_lwt":
            self.plotting_range = {
                "eir-f-t": {
                    "x1_min": lwt,
                    "x1_max": lwt,
                    "x1_norm": lwt,
                    "nbval": nb_val,
                    "x2_min": 10,
                    "x2_max": 60,
                    "x2_norm": lct,
                },
                "cap-f-t": {
                    "x1_min": lwt,
                    "x1_max": lwt,
                    "x1_norm": lwt,
                    "nbval": nb_val,
                    "x2_min": 10,
                    "x2_max": 60,
                    "x2_norm": lct,
                },
                "eir-f-plr": {
                    "x1_min": lct,
                    "x1_max": lct,
                    "x1_norm": lct,
                    "nbval": nb_val,
                    "x2_min": 0.0,
                    "x2_max": 1.0,
                    "x2_norm": 1.0,
                },
            }

        self.ref_lwt, self.ref_ect, self.ref_lct = lwt, ect, lct

    def get_ref_cond_flow_rate(self):
        """Function to compute the reference condenser flow rate given ref_cap, full_eff, ref_lct and ref_lwt

        :return: Reference condenser flow rate
        :rtype: float

        """

        # Convert reference capacity if needed
        if self.ref_cap_unit != "kW":
            evap_cap_ton = Units(value=self.ref_cap, unit=self.ref_cap_unit)
            evap_power = evap_cap_ton.conversion(new_unit="kW")
        else:
            evap_power = self.ref_cap

        if self.ref_cap_unit != "kW/ton":
            ref_cap = Units(value=self.ref_cap, unit=self.ref_cap_unit)
            ref_cap = ref_cap.conversion(new_unit="kW/ton")
        else:
            ref_cap = self.ref_cap

        # Convert reference efficiency if needed
        if self.full_eff_unit != "kW/ton":
            full_eff_unit = Units(value=self.full_eff, unit=self.full_eff_unit)
            full_eff = full_eff_unit.conversion(
                new_unit="kW/ton"
            )  # full eff needs to be in kW/ton
        else:
            full_eff = self.full_eff

        # Retrieve curves
        curves = self.get_chiller_curves()
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_plr = curves["eir_f_plr"]

        cap_f_lwt_lct_rated = cap_f_t.evaluate(self.ref_lwt, self.ref_lct)
        cap_f_lwt_lct = cap_f_t.evaluate(self.ref_lwt, self.ref_lct)
        cap_op = 1.0 * cap_f_lwt_lct
        plr = 1.0 * cap_f_lwt_lct_rated / cap_op

        # Calculate compressor power [kW]
        comp_power = (
            ref_cap
            * full_eff
            * cap_f_lwt_lct
            * eir_f_t.evaluate(self.ref_lwt, self.ref_lct)
            * eir_f_plr.evaluate(self.ref_lct, plr)
        )
        cond_cap = evap_power + comp_power

        # Determine the specific heat capacity of water [kJ/kg.K]
        c_p = (
            CP.PropsSI(
                "C",
                "P",
                101325,
                "T",
                0.5 * (self.ref_ect + self.ref_lct) + 273.15,
                "Water",
            )
            / 1000
        )

        # Determine density of water [kg/m3]
        rho = CP.PropsSI(
            "D", "P", 101325, "T", 0.5 * (self.ref_ect + self.ref_lct) + 273.15, "Water"
        )

        # Determine condenser flow rate at reference conditions [m3/s]
        ref_cond_flow_rate = cond_cap / ((self.ref_lct - self.ref_ect) * c_p * rho)

        return ref_cond_flow_rate

    def calc_eff_ect(self, cap_f_t, eir_f_t, eir_f_plr, eir_ref, ect, lwt, load):
        """Calculate chiller efficiency using the ECT-based model for a specific ECT and load value (percentage load).

        :param Curve cap_f_t: Capacity curve modifier as a function of temperature (LWT and ECT)
        :param Curve eir_f_t: Energy Input Ratio curve modifier as a function of temperature (LWT and ECT)
        :param Curve eir_f_plr: Energy Input Ratio curve modifier as a function of part load ratio
        :param float eir_ref: Reference EIR
        :param float ect: Entering condenser temperature in deg. C
        :param float lwt: Leaving water temperature in deg. C
        :param float load: Percentage load, as defined in AHRI 550/590

        """
        # Temperature adjustments
        dt = ect - lwt
        cap_f_lwt_ect = cap_f_t.evaluate(lwt, ect)
        eir_f_lwt_ect = eir_f_t.evaluate(lwt, ect)
        cap_op = cap_f_lwt_ect

        # PLR adjustments
        plr = load * cap_f_t.evaluate(lwt, self.ref_ect) / cap_op
        if plr <= self.min_unloading:
            plr = self.min_unloading
        eir_plr = eir_f_plr.evaluate(plr, dt)

        # Efficiency calculation
        eir = eir_ref * eir_f_lwt_ect * eir_plr / plr

        return eir

    def calc_rated_eff(self, eff_type, unit="kW/ton", output_report=False, alt=False):
        """Calculate chiller efficiency.

        :param str eff_type: Chiller efficiency type, currently supported `full` (full load rating)
                             and `part` (part load rating)
        :param str unit: Efficiency unit
        :return: Chiller rated efficiency
        :rtype: float

        """
        # Get reference eir
        eir_ref = self.get_eir_ref(alt)
        load_ref = 1

        # Rated conditions as per AHRI Std 551/591
        loads = [1, 0.75, 0.5, 0.25]

        # List of equipment efficiency for each load
        kwpton_lst = []

        # Temperatures at rated conditions
        ect, lwt, lct = self.get_rated_temperatures(alt)

        # Retrieve curves
        curves = self.get_chiller_curves()
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_plr = curves["eir_f_plr"]

        try:
            for idx, load in enumerate(
                loads
            ):  # Calculate efficiency for each testing conditions
                if self.model == "ect_lwt":  # DOE-2 chiller model
                    # Efficiency calculation
                    eir = self.calc_eff_ect(
                        cap_f_t, eir_f_t, eir_f_plr, eir_ref, ect[idx], lwt, load
                    )

                elif self.model == "lct_lwt":  # Reformulated EIR chiller model
                    # Determine water properties
                    c_p = (
                        CP.PropsSI("C", "P", 101325, "T", ect[idx] + 273.15, "Water")
                        / 1000
                    )  # kJ/kg.K
                    rho = CP.PropsSI("D", "P", 101325, "T", ect[idx] + 273.15, "Water")

                    # Gather arguments for determination fo leaving condenser temperature through iteration
                    if idx == 0:  # Full load rated conditions
                        args = [
                            lwt,
                            cap_f_t,
                            eir_f_t,
                            eir_f_plr,
                            load,
                            -999,
                            1 / eir_ref,
                            ect[idx],
                            self.set_of_curves[0].ref_cond_fluid_flow * rho,
                            c_p,
                        ]
                    else:
                        args = [
                            lwt,
                            cap_f_t,
                            eir_f_t,
                            eir_f_plr,
                            load,
                            cap_f_lwt_lct_rated,
                            1 / eir_ref,
                            ect[idx],
                            self.set_of_curves[0].ref_cond_fluid_flow * rho,
                            c_p,
                        ]

                    # Determine leaving condenser temperature
                    lct = self.get_lct(ect[idx], args)

                    # Determine rated capacity curve modifier
                    if idx == 0:
                        cap_f_lwt_lct_rated = cap_f_t.evaluate(lwt, lct)

                    # Temperature adjustments
                    dt = ect[idx] - lwt
                    cap_f_lwt_lct = cap_f_t.evaluate(lwt, lct)
                    eir_f_lwt_lct = eir_f_t.evaluate(lwt, lct)
                    cap_op = load_ref * cap_f_lwt_lct

                    # PLR adjustments
                    plr = load * cap_f_lwt_lct_rated / cap_op
                    if plr <= self.min_unloading:
                        plr = self.min_unloading
                    eir_plr_lct = eir_f_plr.evaluate(lct, plr)

                    # Efficiency calculation
                    eir = eir_ref * eir_f_lwt_lct * eir_plr_lct / plr

                    if eir < 0:
                        return 999

                else:
                    return -999

                # Convert efficiency to kW/ton
                eir = Units(eir, "eir")
                kwpton = eir.conversion("kW/ton")

                if output_report:
                    cap_ton = self.ref_cap
                    if self.ref_cap_unit != "ton":
                        cap_ton = Units(self.ref_cap, self.ref_cap_unit).conversion(
                            "ton"
                        )
                    part_report = f"""At {str(round(load * 100.0, 0)).replace('.0', '')}% load and AHRI rated conditions:
                    - Entering condenser temperature: {round(ect[idx], 2)},
                    - Leaving chiller temperature: {round(lwt, 2)},
                    - Part load ratio: {round(plr, 2)},
                    - Operating capacity: {round(cap_op * cap_ton, 2)} ton,
                    - Power: {round(kwpton * cap_op * cap_ton, 2)} kW,
                    - Efficiency: {round(kwpton, 3)} kW/ton
                    """
                    logging.info(part_report)

                # Store efficiency for IPLV calculation
                kwpton_lst.append(kwpton)

                # Stop here for full load calculations
                if eff_type == "full" and idx == 0:
                    if unit != "kW/ton":
                        kwpton = Units(kwpton, "kW/ton").conversion(unit)
                    return kwpton

            # Coefficients from AHRI Std 551/591
            iplv = 1 / (
                (0.01 / kwpton_lst[0])
                + (0.42 / kwpton_lst[1])
                + (0.45 / kwpton_lst[2])
                + (0.12 / kwpton_lst[3])
            )

            if output_report:
                logging.info(f"IPLV: {round(iplv, 3)} kW/ton")
        except:
            return -999

        # Convert IPLV to desired unit
        if unit != "kW/ton":
            iplv_org = Units(iplv, "kW/ton")
            iplv = iplv_org.conversion(unit)

        return iplv

    def get_rated_temperatures(self, alt=False):
        """Get chiller rated temperatures.

        :param bool alt: Indicate the chiller alternate standard rating should be used
        :return: Rated entering condenser temperature and leaving chilled water temperature
        :rtype: list

        """
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std
        chiller_data = equipment_references[self.type][std][self.condenser_type]
        lwt = Equipment.convert_to_deg_c(chiller_data["lwt"], chiller_data["lwt_unit"])
        ect = [
            Equipment.convert_to_deg_c(t, chiller_data["ect_unit"])
            for t in chiller_data["ect"]
        ]
        lct = Equipment.convert_to_deg_c(chiller_data["lct"], chiller_data["lct_unit"])
        return [ect, lwt, lct]

    def get_chiller_curves(self):
        """Retrieve chiller curves from the chiller set_of_curves attribute.

        :return: Dictionary of the curves associated with the chiller object
        :rtype: dict

        """
        curves = {}
        for curve in self.set_of_curves:
            if curve.out_var == "cap-f-t":
                curves["cap_f_t"] = curve
            elif curve.out_var == "eir-f-t":
                curves["eir_f_t"] = curve
            else:
                curves["eir_f_plr"] = curve

        return curves

    def get_lct(self, ect, args):
        """Determine a chiller's leaving condenser temperature based on a entering condenser temperature.
        The temperature should be determine by iteration, a `secant` root finding methods is used.

        :param float ect: Entering condenser temperature (deg. C)
        :return: Leaving condenser temperature (deg. C)
        :rtype: float

        """
        result = optimize.root_scalar(
            self.cond_inlet_temp_residual,
            args=(args),
            method="secant",
            x0=ect + 0.1,
            x1=ect + 10,
            rtol=0.001,
        )
        lct = result.root
        return lct

    def cond_inlet_temp_residual(self, lct, args):
        """Calculate the enetering condenser temperature residual based on a leaving condenser temperature.

        :param float lct: Leaving condenser temperature (deg. C)
        :return: Entering condenser temperature residual
        :rtype: float

        """
        # Get arguments
        (
            lwt,
            cap_f_t,
            eir_f_t,
            eir_f_plr,
            load,
            cap_f_lwt_lct_rated,
            ref_cop,
            ect,
            m_c,
            c_p,
        ) = args

        # Temperature dependent curve modifiers
        cap_f_lwt_lct = cap_f_t.evaluate(lwt, lct)
        eir_f_lwt_lct = eir_f_t.evaluate(lwt, lct)

        # Convert reference capacity to kW
        if self.ref_cap_unit != "kW":
            ref_cap_org = Units(value=self.ref_cap, unit=self.ref_cap_unit)
            ref_cap = ref_cap_org.conversion(new_unit="kW")
        else:
            ref_cap = self.ref_cap

        # Operating variables
        cap_op = ref_cap * cap_f_lwt_lct
        if cap_f_lwt_lct_rated == -999:
            plr = load
        else:
            plr = load * cap_f_lwt_lct_rated / cap_f_lwt_lct

        # PLR and temperature curve modifier
        eir_f_plr_lct = eir_f_plr.evaluate(lct, plr)

        # Evaporator operating capacity
        q_e = cap_op * plr

        # Chiller power
        p = (cap_op / ref_cop) * eir_f_lwt_lct * eir_f_plr_lct

        # Condenser heat transfer
        q_c = p + q_e

        # Store original ECT
        ect_org = ect

        # Recalculate ECT
        ect = lct - q_c / (m_c * c_p)

        return (ect_org - ect) / ect_org

    def get_lib_and_filters(self, lib_path=chiller_lib):
        """Get chiller library object and chiller specific filters.

        :param str lib_path:Full path of json library
        :return: Chiller library object and filters
        :rtype: list

        """
        lib = Library(path=lib_path)
        filters = [
            ("eqp_type", "chiller"),
            ("condenser_type", self.condenser_type),
            ("sim_engine", self.sim_engine),
            ("model", self.model),
        ]

        return lib, filters

    def get_ranges(self):
        """Get applicable range of values for independent variables of the chiller model.

        :return: Range of values, and values used for normalization (reference/rated values)
        :rtype: dict

        """
        norm_val = {"ect_lwt": self.ref_ect, "lct_lwt": self.ref_lct}[self.model]

        ranges = {
            "eir-f-t": {
                "vars_range": [(4, 10), (10.0, 40.0)],
                "normalization": (self.ref_lwt, norm_val),
            },
            "cap-f-t": {
                "vars_range": [(4, 10), (10.0, 40.0)],
                "normalization": (self.ref_lwt, norm_val),
            },
            "eir-f-plr": {"vars_range": [(0.0, 1.0)], "normalization": (1.0)},
        }

        return ranges

    def get_curves_from_lib(self, lib, filters):
        """Function to get the sort from the library based on chiller filters.

        :param copper.library.Library lib: Chiller library object
        :param list filters: List of tuples containing the relevant filter keys and values
        :return: List of set of curves object corresponding to seed curves
        :rtype: list

        """

        # add a block for constant + variable compressor speed
        if self.compressor_speed in ["constant", "variable"]:
            filters = filters + [("compressor_speed", self.compressor_speed)]
        elif self.compressor_speed == "any":
            filters = filters

        # Selecting the relevant filters based on compressor type
        if self.compressor_type == "positive_displacement":
            tr_wtr_screw = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "screw")], part_eff_flag=True
            )
            tr_wtr_scroll = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "scroll")], part_eff_flag=True
            )
            tr_wtr_rec = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "reciprocating")],
                part_eff_flag=True,
            )
            sets = tr_wtr_screw + tr_wtr_scroll + tr_wtr_rec
        elif self.compressor_type == "any":
            tr_wtr_screw = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "screw")], part_eff_flag=True
            )
            tr_wtr_scroll = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "scroll")], part_eff_flag=True
            )
            tr_wtr_rec = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "reciprocating")],
                part_eff_flag=True,
            )
            ep_wtr_cent = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "centrifugal")],
                part_eff_flag=True,
            )
            sets = tr_wtr_screw + tr_wtr_scroll + tr_wtr_rec + ep_wtr_cent
        elif self.compressor_type == "scroll/screw":
            tr_wtr_screw = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "screw")], part_eff_flag=True
            )
            tr_wtr_scroll = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", "scroll")], part_eff_flag=True
            )
            sets = tr_wtr_screw + tr_wtr_scroll
        elif self.compressor_type in [
            "scroll",
            "screw",
            "reciprocating",
            "centrifugal",
        ]:
            ep_wtr = lib.find_set_of_curves_from_lib(
                filters=filters + [("compressor_type", self.compressor_type)],
                part_eff_flag=True,
            )
            sets = ep_wtr
        else:
            sets = None

        assert sets is not None

        return sets

    def get_seed_curves(self, lib=None, filters=None, csets=None):
        """Function to generate seed curves specific to a chiller and sets relevant attributes (misc_attr, ranges).

        :param copper.library.Library lib: Chiller library object
        :param list fitlers: List of tuples containing the filter keys and values
        :param list csets: List of set of curves object corresponding to selected chillers from library
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
            "ref_cap": self.ref_cap,
            "ref_cap_unit": "",
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
