class equipment:
    def __init__(
        self,
        eqp_type,
        eqp_subtype,
        eqp_subtype_1,
        sim_engine,
        algorithm,
        full_eff,
        full_eff_units,
        part_eff,
        part_eff_units,
        nb_curves,
        precision,
    ):
        self.eqp_type = eqp_type
        self.eqp_subtype = eqp_subtype
        self.eqp_subtype = eqp_subtype_1
        self.sim_engine = sim_engine
        self.algorithm = algorithm
        self.full_eff = full_eff
        self.full_eff_units = full_eff_units
        self.part_eff = part_eff
        self.part_eff_units = part_eff_units
        self.nb_curves = nb_curves
        self.precision = precision

    def gen_perf_curves(self):
        return True

    def plot_perf_curves(self):
        return True


class rating:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_eff_rat):
        if new_eff_rat == "kWpton":
            if self.unit == "COP":
                return 12.0 / (self.value * 3.412)
            elif self.unit == "kWpton":
                return self.value
            elif self.unit == "EER":
                return 12.0 / self.value
            else:
                return self.value
        elif new_eff_rat == "COP":
            if self.unit == "kWpton":
                return 12.0 / self.value / 3.412
            elif self.unit == "COP":
                return self.value
            elif self.unit == "EER":
                return self.value / 3.412
            else:
                return self.value
        elif new_eff_rat == "EER":
            if self.unit == "kWpton":
                return 12.0 / self.value
            elif self.unit == "EER":
                return self.value
            elif self.unit == "COP":
                return 3.412 * self.value
            else:
                return self.value
