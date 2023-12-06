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

location = os.path.dirname(os.path.realpath(__file__))
unitary_dx_lib = os.path.join(
    location, "data", "unitary_dx_curves.json"
)  # TODO: add the library file.
equipment_references = json.load(
    open(os.path.join(location, "data", "equipment_references.json"), "r")
)


class Unitarydx(Equipment):
    def __init__(
        self,
        ref_cap,
        ref_cap_unit,
        full_eff,
        full_eff_unit,
        compressor_type,
        condenser_type,
        compressor_speed,
        part_eff = 0,
        part_eff_unit = "",
        set_of_curves="",
        part_eff_ref_std="ahri_340/360",
        part_eff_ref_std_alt ="ahri_341/361",
        model="simplified_bf", #what is this?
        sim_engine="energyplus",
    ):
        self.type = "unitary_dx"
        self.ref_cap = ref_cap
        self.ref_cap_unit = ref_cap_unit
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
        self.condenser_type = "air"
        # Define rated temperatures
        # air entering drybulb,air entering wetbulb, outdoor enter, outdoor leaving
        AED, AEW, ect, lct = self.get_rated_temperatures()
        ect = ect[0]

        # Defined plotting ranges and (rated) temperature for normalization
        nb_val = 50
        if self.model == "X":
            self.plotting_range = {
                "eir-f-t": {
                    "x1_min": AEW,
                    "x1_max": AEW,
                    "x1_norm": AEW,
                    "nbval": nb_val,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "cap-f-t": {
                    "x1_min": AEW,
                    "x1_max": AEW,
                    "x1_norm": AEW,
                    "nbval": 50,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "eir-f-f": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
                "cap-f-f": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
                "plf-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
            }


    def calc_rated_eff(self, eff_type, unit="eer", output_report=False, alt=False):
        """Calculate unitary DX equipment efficiency.

        :param str eff_type: Unitary DX equipment efficiency type, currently supported `full` (full load rating)
                             and `part` (part load rating)
        :param str unit: Efficiency unit
        :return: Unitary DX Equipment rated efficiency
        :rtype: float

        """
        # Get reference eir
        eir_ref = self.get_eir_ref(alt)
        load_ref = 1

        # Rated conditions as per AHRI Std 340/360
        loads = [1, 0.75, 0.5, 0.25]

        # List of equipment efficiency for each load
        kwpton_lst = []

        # Temperatures at rated conditions
        AED, AEW, ect, lct = self.get_rated_temperatures(alt)

        # Retrieve curves
        #To DO check curvetypes
        curves = self.get_DX_curves()
        cap_f_f = curves["cap_f_f"]
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_f = curves["eir_f_f"]
        plf_f_plr = curves["plf_f_plr"]
        
        #TODO I/O process
        try:
            for idx, load in enumerate(
                loads
            ):  # Calculate efficiency for each testing conditions
                if self.model == "ect_lwt":  # DOE-2 chiller model
                    # Efficiency calculation
                    eir = self.calc_eff_ect(
                        cap_f_t, eir_f_t, cap_f_f, eir_ref, ect[idx], lwt, load
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
                            eir_f_f,
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
                            eir_f_f,
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
                    eir_plr_lct = eir_f_f.evaluate(lct, plr)

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

    def get_DX_curves(self):
        """Retrieve DX curves from the DX set_of_curves attribute.

        :return: Dictionary of the curves associated with the object
        :rtype: dict

        """
        curves = {}
        for curve in self.set_of_curves:
            if curve.out_var == "cap-f-t":
                curves["cap_f_t"] = curve
            elif curve.out_var == "cap-f-f":
                curves["cap_f_f"] = curve
            elif curve.out_var == "eir-f-t":
                curves["eir_f_t"] = curve
            elif curve.out_var == "eir-f-f":
                curves["eir_f_f"] = curve
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
        DX_data = equipment_references[self.type][std][self.condenser_type]
        # Air Entering Indoor Drybulb
        AED = [
            Equipment.convert_to_deg_c(t, DX_data["AE_unit"])
            for t in DX_data["AED"]
        ]
        # Air Entering Indoor Wetbulb
        AEW = [
            Equipment.convert_to_deg_c(t, DX_data["AE_unit"])
            for t in DX_data["AEW"]
        ]
        #Outdoor Water/Air entering
        ect = [
            Equipment.convert_to_deg_c(t, DX_data["ect_unit"])
            for t in DX_data["ect"]
        ]
        #Outdoor Water/Air leaving
        lct = Equipment.convert_to_deg_c(DX_data["lct"], DX_data["lct_unit"])
        return [AED, AEW, ect, lct]
    
    def get_lib_and_filters(self, lib_path=unitary_dx_lib):
        """Get unitary DX equipment library object and unitary DX equipment specific filters.

        :param str lib_path:Full path of json library
        :return: Unitary DX equipment library object and filters
        :rtype: list

        """
        lib = Library(path=lib_path)
        filters = [
            ("eqp_type", "UnitaryDX"),
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
        :param list fitlers: List of tuples containing the filter keys and values
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
    
    def calc_eff_ect(self, cap_f_t, eir_f_t, eir_f_plr, eir_ref, ect, lwt, load):
        """Calculate DX system efficiency

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
        curves = self.get_DX_curves()
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_plr = curves["eir_f_plr"]

        cap_f_lwt_lct_rated = cap_f_t.evaluate(self.ref_AEW, self.ref_ect)
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

if __name__=="__main__":
    lib = Library(path=unitary_dx_lib)
    X = Unitarydx(compressor_type="centrifugal",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap=471000,
            ref_cap_unit="W",
            full_eff=5.89,
            full_eff_unit="cop",
            part_eff_ref_std="ahri_340/360",
            model="X",
            sim_engine="energyplus",
            set_of_curves=lib.get_set_of_curves_by_name("0").curves,)
    
    #
    #print(lib.get_set_of_curves_by_name("0").curves)

    print(X.get_ref_values("cap-f-t"))
    #print(X.get_ref_cond_flow_rate())
