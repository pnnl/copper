"""
units.py
====================================
The conversion module of Copper handles simple unit conversions to avoid creating additional dependencies.
"""


class Units:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_unit):
        """Convert efficiency rating.

        :param str new_unit: Unit after conversion
        :return: Converted value
        :rtype: float

        """
        ton_to_kbtu = 12
        kbtu_to_kw = 3.412141633
        if new_unit == "kW/ton":
            if self.unit == "cop":
                return ton_to_kbtu / (self.value * kbtu_to_kw)
            elif self.unit == "kW/ton":
                return self.value
            elif self.unit == "eer":
                return ton_to_kbtu / self.value
            elif self.unit == "eir":
                return ton_to_kbtu * self.value / kbtu_to_kw
            else:
                return self.value
        elif new_unit == "cop":
            if self.unit == "kW/ton":
                return ton_to_kbtu / self.value / kbtu_to_kw
            elif self.unit == "cop":
                return self.value
            elif self.unit == "eer":
                return self.value / kbtu_to_kw
            elif self.unit == "eir":
                return 1 / self.value
            else:
                return self.value
        elif new_unit == "eer":
            if self.unit == "kW/ton":
                return ton_to_kbtu / self.value
            elif self.unit == "eer":
                return self.value
            elif self.unit == "cop":
                return kbtu_to_kw * self.value
            elif self.unit == "eir":
                return kbtu_to_kw / self.value
            else:
                return self.value
        elif new_unit == "eir":
            if self.unit == "kW/ton":
                return 1 / (ton_to_kbtu / self.value / kbtu_to_kw)
            elif self.unit == "eer":
                return kbtu_to_kw / self.value
            elif self.unit == "cop":
                return 1 / self.value
            elif self.unit == "eir":
                return self.value
            else:
                return self.value
        elif new_unit == "ton":
            if self.unit == "kW":
                return self.value * (kbtu_to_kw / ton_to_kbtu)
            elif self.unit == "W":
                return self.value * (kbtu_to_kw / (ton_to_kbtu * 1000))
            if self.unit == "ton":
                return self.value
        elif new_unit == "kW":
            if self.unit == "ton":
                return self.value / (kbtu_to_kw / ton_to_kbtu)
            if self.unit == "W":
                return self.value / 1000
            if self.unit == "kW":
                return self.value
        elif new_unit == "W":
            if self.unit == "ton":
                return self.value / (kbtu_to_kw / (ton_to_kbtu * 1000))
            if self.unit == "kW":
                return self.value * 1000
            if self.unit == "W":
                return self.value
        elif new_unit == "degC":
            if self.unit == "degF":
                return (self.value - 32) * 5 / 9
        elif new_unit == "degF":
            if self.unit == "degC":
                return (self.value * 9 / 5) + 32
