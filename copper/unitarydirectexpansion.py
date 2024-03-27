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
    location, "data", "unitarydirectexpansion_curves.json"
)
equipment_references = json.load(
    open(os.path.join(location, "data", "equipment_references.json"), "r")
)


class UnitaryDirectExpansion(Equipment):
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
        set_of_curves="",
        part_eff_ref_std="ahri_340/360",
        part_eff_ref_std_alt=None,
        model="simplified_bf",
        sim_engine="energyplus",
    ):
        self.type = "UnitaryDirectExpansion"
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
        FanPower = 7000 #Q1: how to deal with fan power? place holder for fan power Btu/h
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std
        # Get reference eir
        eir_ref = Equipment.get_eir_ref(alt)
        load_ref = 1

        # List of equipment efficiency for each load
        kwpton_lst = []

        # Temperatures at rated conditions
        aed, aew, ect, lct = self.get_rated_temperatures(alt)
        # Retrieve curves
        curves = self.get_DX_curves()
        cap_f_f = curves["cap_f_ff"]
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_f = curves["eir_f_ff"]
        plf_f_plr = curves["plf_f_plr"]
        cap_f_aew_ect = [cap_f_t.evaluate(aew[0], item) for item in ect]
        eir_f_aew_ect = [eir_f_t.evaluate(aew[0], item) for item in ect]
        eir_f_ff = eir_f_f.evaluate(1, 1)
        cap_f_ff = cap_f_f.evaluate(1, 1)
        eir = [eir_ref * item * eir_f_ff for item in eir_f_aew_ect]
        for i in range(0, 4):
            x = Units(eir[i], "eir")
            eir[i] = x.conversion("kW/ton")
        NumOfReducedCap = equipment_references[self.type][std]['coef']['NumOfReducedCap']
        ReducedPLR = equipment_references[self.type][std]['coef']['ReducedPLR']
        WeightingFactor = equipment_references[self.type][std]['coef']['WeightingFactor']
        NetCoolingCapRated = self.ref_cap # it may not be correct
        for RedCapNum in range(NumOfReducedCap):
            """
            Q2: do we need this part? I dont see reduced test condition in the standard 340
            if ReducedPLR[RedCapNum] > 0.444:
                OutdoorUnitInletAirDryBulbTempReduced = 5.0 + 30.0 * ReducedPLR[RedCapNum]
            else:
                OutdoorUnitInletAirDryBulbTempReduced = OADBTempLowReducedCapacityTest
            """
            NetCoolingCapReduced = self.ref_cap*cap_f_aew_ect[RedCapNum]*cap_f_ff[RedCapNum] - FanPower
            LoadFactor = ReducedPLR[RedCapNum] * NetCoolingCapRated / NetCoolingCapReduced if NetCoolingCapReduced > 0.0 else 1.0
            DegradationCoeff = 1.130 - 0.130 * LoadFactor
            ElecPowerReducedCap = DegradationCoeff * eir[RedCapNum] * (self.ref_cap * cap_f_aew_ect[RedCapNum]*cap_f_ff[RedCapNum])
            EERReduced = (LoadFactor * NetCoolingCapReduced) / (LoadFactor * ElecPowerReducedCap + FanPower)
            IEER = IEER + WeightingFactor[RedCapNum] * EERReduced
        # note eir = 1/COP, EER = COP*3.413
        return IEER

    def get_DX_curves(self):
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
        DX_data = equipment_references[self.type][std][self.condenser_type]
        # Air Entering Indoor Drybulb
        aed = [
            Equipment.convert_to_deg_c(t, DX_data["ae_unit"]) for t in DX_data["aed"]
        ]
        # Air Entering Indoor Wetbulb
        aew = [
            Equipment.convert_to_deg_c(t, DX_data["ae_unit"]) for t in DX_data["aew"]
        ]
        # Outdoor Water/Air entering
        ect = [
            Equipment.convert_to_deg_c(t, DX_data["ect_unit"]) for t in DX_data["ect"]
        ]
        # Outdoor Water/Air leaving
        lct = Equipment.convert_to_deg_c(DX_data["lct"], DX_data["lct_unit"])
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
