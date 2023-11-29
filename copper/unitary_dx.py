"""
unitary_dx.py
====================================
This is the unitary direct expansion (DX) HVAC equipment module of Copper. The module handles all calculations and data manipulation related to unitary DX equipment.
"""

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


class UnitaryDX(Equipment):
    def __init__(
        self,
        ref_cap,
        ref_cap_unit,
        full_eff,
        full_eff_unit,
        part_eff,
        part_eff_unit,
        compressor_type,
        set_of_curves,
        part_eff_ref_std="ahri_340/360",
        part_eff_ref_std_alt ="ahri_341/361",
        model="simplified_bf",
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
        ect, lwt, lct = self.get_rated_temperatures(alt)

        # Retrieve curves
        curves = self.get_DX_curves()
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_plr = curves["eir_f_plr"]

        #TODO I/O process

    def get_DX_curves(self):
        """Retrieve DX curves from the DX set_of_curves attribute.

        :return: Dictionary of the curves associated with the object
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
            ("eqp_type", "DX"),
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
            "cap-f-t": {
                "vars_range": [(4, 10), (10.0, 40.0)],
                "normalization": (self.ref_lwt, norm_val),
            },
            "eir-f-plr": {"vars_range": [(0.0, 1.0)], "normalization": (1.0)},
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
