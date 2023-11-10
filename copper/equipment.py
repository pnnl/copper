"""
equipment.py
====================================
This is the equipment module of Copper. The module includes function that can be used by all types of equipment included in Copper.
"""

from copper.units import *


class Equipment:
    def __init__(self):
        self.plotting_range = {}
        self.full_eff_alt = None
        self.full_eff_unit_alt = None
        self.full_eff = None
        self.full_eff_unit = None

    def convert_to_deg_c(value, unit="degF"):
        """Helper function to convert equipment data to degree F.

        :param float value: Value to convert to degree C
        :return: Vlue converted to degree C
        :rtype: float
        """
        if unit == "degF":
            curr_value = Units(value, unit)
            return curr_value.conversion("degC")
        else:
            return value

    def get_ref_values(self, out_var):
        """Get equipment reference/rated independent variables values (temperature and part load ratio) for an output variable (e.g., eir-f-t, eir-f-plr, cap-f-t).

        :param str out_var: Output variable
        :return: List of reference values
        :rtype: list

        """
        if "x2_norm" in list(self.plotting_range[out_var].keys()):
            return [
                self.plotting_range[out_var]["x1_norm"],
                self.plotting_range[out_var]["x2_norm"],
            ]
        else:
            return [self.plotting_range[out_var]["x1_norm"], 0.0]

    def get_eir_ref(self, alt):
        """Get the reference EIR (energy input ratio) of an equipment.

        :param bool alt: Specify if the alternative equipment efficiency should be used to calculate the EIR
        :return: Reference EIR
        :rtype: float

        """
        # Retrieve equipment efficiency and unit
        if alt:
            ref_eff = self.full_eff_alt
            ref_eff_unit = self.full_eff_unit_alt
        else:
            ref_eff = self.full_eff
            ref_eff_unit = self.full_eff_unit
        eff = Units(ref_eff, ref_eff_unit)
        return eff.conversion("eir")
