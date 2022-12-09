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
    def cop_to_eer(cls, input_value: float) -> float:
        return cls.COP_CONSTANT * input_value

    @classmethod
    def kw_per_ton_to_cop(cls, input_value: float) -> float:
        return cls.EER / input_value / cls.COP_CONSTANT

    @classmethod
    def kw_per_ton_to_eer(cls, input_value: float) -> float:
        return cls.EER / input_value

    @classmethod
    def eer_to_cop(cls, input_value: float) -> float:
        return input_value / cls.COP_CONSTANT

    @classmethod
    def eer_to_kw_per_ton(cls, input_value: float) -> float:
        return cls.EER / input_value

    @classmethod
    def kw_to_ton(cls, input_value: float) -> float:
        return input_value * (cls.COP_CONSTANT / cls.EER)

    @classmethod
    def watt_to_ton(cls, input_value: float) -> float:
        return input_value * (cls.COP_CONSTANT / (cls.EER * cls.KILO))

    @classmethod
    def ton_to_kw(cls, input_value: float) -> float:
        return input_value / (cls.COP_CONSTANT / cls.EER)

    @classmethod
    def watt_to_kw(cls, input_value: float) -> float:
        return input_value / cls.KILO

    @classmethod
    def ton_to_watt(cls, input_value: float) -> float:
        return input_value / (cls.COP_CONSTANT / (cls.EER * cls.KILO))

    @classmethod
    def F_to_C(cls, input_value: float) -> float:
        return (input_value - cls.TEMP_CONSTANT) * cls.FIVE_OVER_NINE

    @classmethod
    def C_to_F(cls, input_value: float) -> float:
        return (input_value * cls.NINE_OVER_FIVE) + cls.TEMP_CONSTANT

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
            if self.unit == "cop":                      # done
                return 12.0 / (self.value * 3.412) 
            elif self.unit == "kw/ton":                 # done
                return self.value
            elif self.unit == "eer":                    # done
                return 12.0 / self.value
            else:
                return self.value
        elif new_unit == "cop":
            if self.unit == "kw/ton":                   # done
                return 12.0 / self.value / 3.412 
            elif self.unit == "cop":                    # done
                return self.value
            elif self.unit == "eer":                    # done
                return self.value / 3.412
            else:
                return self.value
        elif new_unit == "eer":
            if self.unit == "kw/ton":                   # done
                return 12.0 / self.value
            elif self.unit == "eer":                    # done
                return self.value
            elif self.unit == "cop":                    # done
                return 3.412 * self.value
            else:
                return self.value
        elif new_unit == "ton":
            if self.unit == "kW":                       # done
                return self.value * (3412 / 12000)
            elif self.unit == "W":                      # done
                return self.value * (3.412 / 12000)
        elif new_unit == "kW":
            if self.unit == "ton":                      # done
                return self.value / (3412 / 12000)
            if self.unit == "W":                        # done
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
