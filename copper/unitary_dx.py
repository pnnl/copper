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
        model="simplified_bf",
        sim_engine="energyplus",
    ):
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

    def calc_rated_eff(self, eff_type, unit="eer", output_report=False, alt=False):
        """Calculate unitary DX equipment efficiency.

        :param str eff_type: Unitary DX equipment efficiency type, currently supported `full` (full load rating)
                             and `part` (part load rating)
        :param str unit: Efficiency unit
        :return: Unitary DX Equipment rated efficiency
        :rtype: float

        """
        pass

    def get_rated_temperatures(self, alt=False):
        """Get unitary DX equipment rated temperatures.

        :param bool alt: Indicate the unitary DX equipment alternate standard rating should be used
        :return: Rated entering condenser temperature and temperature of the air entering the system
        :rtype: list

        """

    def get_lib_and_filters(self, lib_path=unitary_dx_lib):
        """Get unitary DX equipment library object and unitary DX equipment specific filters.

        :param str lib_path:Full path of json library
        :return: Unitary DX equipment library object and filters
        :rtype: list

        """
        pass

    def get_ranges(self):
        """Get applicable range of values for independent variables of the unitary DX equipment model.

        :return: Range of values, and values used for normalization (reference/rated values)
        :rtype: dict

        """
        pass

    def get_seed_curves(self, lib=None, filters=None, csets=None):
        """Function to generate seed curves specific to a unitary DX equipment and sets relevant attributes (misc_attr, ranges).

        :param copper.library.Library lib: Unitary DX equipment library object
        :param list fitlers: List of tuples containing the filter keys and values
        :param list csets: List of set of curves object corresponding to selected unitary DX equipment from library
        :rtype: SetsofCurves

        """
        pass
