"""
units.py
====================================
The conversion module of Copper handles simple unit conversions to avoid creating additional dependencies.
"""
from enum import Enum, auto

class ConversionNames(Enum):
    COP = "cop"
    KW_PER_TON = "kw/ton"
    EER = "eer"
    TON = "ton"
    W = "W"
    kW = "kW"
    F_TO_C = "degF"
    C_TO_F = "degC"

class updatedUnits:
    COP_CONSTANT = 3.412
    EER = 12.0
    KILO = 1000
    kEER = EER * KILO
    FIVE_OVER_NINE = 5/9
    NINE_OVER_FIVE = FIVE_OVER_NINE ** -1
    TEMP_CONSTANT = 32
    
    @classmethod
    def cop_to_kw_per_ton(cls, input_value: float) -> float:
        return cls.EER / (input_value * cls.COP_CONSTANT)

    @classmethod
    def kw_per_ton_to_cop(cls, input_value:float) -> float:
        return cls.EER / input_value / cls.COP_CONSTANT

    @classmethod
    def eer_to_kw_per_ton(cls, input_value: float) -> float:
        return cls.EER / input_value

    @classmethod
    def kw_per_ton_to_eer(cls, input_value: float) -> float:
        return cls.eer_to_kw_per_ton(input_value) ** -1

    @classmethod
    def eer_to_cop(cls, eer_value):
        return eer_value / cls.COP_CONSTANT

class Units:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_unit: str) -> float:
        """Convert efficiency rating.

        :param str new_unit: Unit after conversion
        :return: Converted value
        :rtype: float

        """
        if new_unit == "kw/ton":
            if self.unit == "cop":
                return 12.0 / (self.value * 3.412) # done
            elif self.unit == "kw/ton":
                return self.value
            elif self.unit == "eer":
                return 12.0 / self.value # done
            else:
                return self.value
        elif new_unit == "cop":
            if self.unit == "kw/ton":
                return 12.0 / self.value / 3.412 # done
            elif self.unit == "cop":
                return self.value
            elif self.unit == "eer":
                return self.value / 3.412 # done
            else:
                return self.value
        elif new_unit == "eer":
            if self.unit == "kw/ton":
                return 12.0 / self.value
            elif self.unit == "eer":
                return self.value
            elif self.unit == "cop":
                return 3.412 * self.value
            else:
                return self.value
        elif new_unit == "ton":
            if self.unit == "kW":
                return self.value * (3412 / 12000)
            elif self.unit == "W":
                return self.value * (3.412 / 12000)
        elif new_unit == "kW":
            if self.unit == "ton":
                return self.value / (3412 / 12000)
            if self.unit == "W":
                return self.value / 1000
        elif new_unit == "W":
            if self.unit == "ton":
                return self.value / (3.412 / 12000)
            if self.unit == "kW":
                return self.value * 1000
        elif new_unit == "degC":
            if self.unit == "degF":
                return (self.value - 32) * 5 / 9
        elif new_unit == "degF":
            if self.unit == "degC":
                return (self.value * 9 / 5) + 32
