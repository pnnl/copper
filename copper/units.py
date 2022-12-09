"""
units.py
====================================
The conversion module of Copper handles simple unit conversions to avoid creating additional dependencies.
"""
import copper.constants as constants

class Units:
    @classmethod
    def cop_to_kw_per_ton(cls, input_value: float) -> float:
        return constants.TON_TO_KBTU / (input_value * constants.KBTU_TO_KW)

    @classmethod
    def cop_to_eer(cls, input_value: float) -> float:
        return constants.KBTU_TO_KW * input_value

    @classmethod
    def kw_per_ton_to_cop(cls, input_value: float) -> float:
        return constants.TON_TO_KBTU / input_value / constants.KBTU_TO_KW

    @classmethod
    def kw_per_ton_to_eer(cls, input_value: float) -> float:
        return constants.TON_TO_KBTU / input_value

    @classmethod
    def eer_to_cop(cls, input_value: float) -> float:
        return input_value / constants.KBTU_TO_KW

    @classmethod
    def eer_to_kw_per_ton(cls, input_value: float) -> float:
        return constants.TON_TO_KBTU / input_value

    @classmethod
    def kw_to_ton(cls, input_value: float) -> float:
        return input_value * (constants.KBTU_TO_KW / constants.TON_TO_KBTU)

    @classmethod
    def watt_to_ton(cls, input_value: float) -> float:
        return input_value * (constants.KBTU_TO_KW / (constants.TON_TO_KBTU * constants.KILO))

    @classmethod
    def ton_to_kw(cls, input_value: float) -> float:
        return input_value / (constants.KBTU_TO_KW / constants.TON_TO_KBTU)

    @classmethod
    def watt_to_kw(cls, input_value: float) -> float:
        return input_value / constants.KILO

    @classmethod
    def ton_to_watt(cls, input_value: float) -> float:
        return input_value / (constants.KBTU_TO_KW / (constants.TON_TO_KBTU * constants.KILO))

    @classmethod
    def F_to_C(cls, input_value: float) -> float:
        return (input_value - constants.TEMP_CONSTANT) * constants.FIVE_OVER_NINE

    @classmethod
    def C_to_F(cls, input_value: float) -> float:
        return (input_value * constants.NINE_OVER_FIVE) + constants.TEMP_CONSTANT

# TODO: replace all the references to the old version of Units
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
