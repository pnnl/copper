import json


class Chiller:
    def __init__(
        self,
        ref_cap,
        ref_cap_unit,
        full_eff,
        full_eff_unit,
        part_eff,
        part_eff_unit,
        compressor_type,
        condenser_type,
        compressor_speed,
    ):
        self.compressor_type = compressor_type
        self.condenser_type = condenser_type
        self.compressor_speed = compressor_speed
        self.ref_cap = ref_cap
        self.ref_cap_unit = ref_cap_unit
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit

    def generate_curve_set(self, sim_engine, algorithm):
        cset = CurveSet(self.compressor_type, [])
        return cset.generate(sim_engine, algorithm)


class CurveSet:
    def __init__(self, name, curves):
        self.curves = curves

    def generate(self, sim_engine, algorithm):
        pass

    def plot(self, unit):
        pass

    def find_curve_sets(
        self,
        db_path="./fixtures/curves.json",
        filters=[],
        features=[],
    ):
        pass
        # curves = json.loads(open(db_path, "r").read())
        # avail_curve_sets = []
        # for curve in curves:
        #    if all(
        #        curves[curve][i] == j for i, j in filters
        #    ):
        #        if features == []:
        #            curve_sets_match = cp.curves_set(curve,)
        #            for curve in curves[curve]["curves"]:
        # return avail_curve_sets


class Curve:
    def __init__(
        self,
        output_var,
        curve_type,
        x_min,
        y_min,
        x_max,
        y_max,
        out_min,
        out_max,
        coeffs,
    ):
        self.output_var = output_var
        self.curve_type = curve_type
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.out_min = out_min
        self.out_max = out_max
        self.coeffs = coeffs

    def evaluate(self, x, y):
        x = min(max(x, self.x_min), self.x_max)
        y = min(max(y, self.y_min), self.y_max)

        if self.curve_type == "bi_quad":
            out = (
                self.coeffs[0]
                + self.coeffs[1] * x
                + self.coeffs[2] * x ** 2
                + self.coeffs[3] * y
                + self.coeffs[4] * y ** 2
                + self.coeffs[5] * x * y
            )
            return min(max(out, self.out_min), self.out_max)
        if self.curve_type == "bi_cub":
            out = (
                self.coeffs[0]
                + self.coeffs[1] * x
                + self.coeffs[2] * x ** 2
                + self.coeffs[3] * y
                + self.coeffs[4] * y ** 2
                + self.coeffs[5] * x * y
                + self.coeffs[6] * x ** 3
                + self.coeffs[7] * y ** 3
                + self.coeffs[8] * y * x ** 2
                + self.coeffs[9] * x * y ** 2
            )
            return min(max(out, self.out_min), self.out_max)
        if self.curve_type == "quad":
            out = (
                self.coeffs[0]
                + self.coeffs[1] * x
                + self.coeffs[2] * x ** 2
            )
            return min(max(out, self.out_min), self.out_max)


class Rating:
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

    def iplv(self, value):
        pass

    def kwpton(self, value):
        pass
