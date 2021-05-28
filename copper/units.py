class Units:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_unit):
        """Convert efficiency rating.

        :param str new_unit: Unit after conversion
        :return: Converted value
        :rtype: int

        """
        if new_unit == "kw/ton":
            if self.unit == "cop":
                return 12.0 / (self.value * 3.412)
            elif self.unit == "kw/ton":
                return self.value
            elif self.unit == "EER":
                return 12.0 / self.value
            else:
                return self.value
        elif new_unit == "cop":
            if self.unit == "kw/ton":
                return 12.0 / self.value / 3.412
            elif self.unit == "cop":
                return self.value
            elif self.unit == "EER":
                return self.value / 3.412
            else:
                return self.value
        elif new_unit == "EER":
            if self.unit == "kw/ton":
                return 12.0 / self.value
            elif self.unit == "EER":
                return self.value
            elif self.unit == "cop":
                return 3.412 * self.value
            else:
                return self.value
        elif new_unit == "ton":
            if self.unit == "kW":
                return self.value * (3412 / 12000)
        elif new_unit == "kW":
            if self.unit == "ton":
                return self.value / (3412 / 12000)
