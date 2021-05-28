import numpy as np
import pandas as pd
import statsmodels.api as sm
import statistics, itertools


class SetsofCurves:
    def __init__(self, eqp_type, sets):
        self.eqp_type = eqp_type
        self.sets_of_curves = sets

    def get_aggregated_set_of_curves(self, method="average", ranges={}, misc_attr={}):
        """
        Determine sets of curves based on aggregation.

        :param string method: Type of aggregation, currently supported: 'average' and 'median.
        :param dict ranges: Dictionary that defines the ranges of values for each independent variable used to calculate aggregated dependent variable values.
        :param dict misc_attr: Dictionary that provides values for the aggregated set of curves.
        :return: Aggregated set of curves.
        :rtype: SetofCurves()
        """
        # Check that all curves are/have:
        # - the same output variables
        # - of the same type
        # - defined for the same type of units

        "first entry -> refsetofcurves"
        ref_setofcurves = self.sets_of_curves[0].list_to_dict()

        for set_of_curves in self.sets_of_curves:
            if set(ref_setofcurves.keys()) != set_of_curves.list_to_dict().keys():
                raise ValueError(
                    "The output variables in each set of curves are not consistently the same, aggregated set of curves cannot currently be determined."
                )
            for c in set_of_curves.curves:
                if c.type != ref_setofcurves[c.out_var].type:
                    raise ValueError(
                        "Curve type in each set of curves are not consistently the same, aggregated set of curves cannot currently be determined."
                    )
                if not c.out_var in list(ranges.keys()):
                    raise ValueError(
                        "Ranges provided do not cover some of the output variables. Ranges: {}, output variables not found in ranges {}.".format(
                            list(ranges.keys()), c.out_var
                        )
                    )
                if c.units != ref_setofcurves[c.out_var].units:
                    raise ValueError(
                        "Curve unit mismatch, aggregated set of curves cannot currently be determined."
                    )

        input_values = {}
        # Determine values of independent variables using the user-specified ranges
        for c in self.sets_of_curves[0].curves:
            input_values[c.out_var] = []
            for vars_rng in ranges[c.out_var]["vars_range"]:
                min_val, max_val = vars_rng
                input_values[c.out_var].append(np.linspace(min_val, max_val, 4))
            # Add 0s for second independent variables for univariate curves
            if len(input_values[c.out_var]) == 1:
                input_values[c.out_var].append(np.linspace(0.0, 0.0, 4))

        output_values = {}
        # Calculate values of dependent variables using the user-specified ranges
        for set_of_curves in self.sets_of_curves:
            for c in set_of_curves.curves:
                output_value = [
                    c.evaluate(x, y)
                    for x, y in list(
                        itertools.product(
                            input_values[c.out_var][0], input_values[c.out_var][1]
                        )
                    )
                ]
                if c.out_var in output_values.keys():
                    output_values[c.out_var].append(output_value)
                else:
                    output_values[c.out_var] = []
                    output_values[c.out_var].append(output_value)

        # Create new set of curves and assign its attributes based on user defined inputs
        agg_set_of_curves = SetofCurves(eqp_type=self.eqp_type)
        for att, att_val in misc_attr.items():
            setattr(agg_set_of_curves, att, att_val)

        # Determine aggregated values for dependent variables
        for var, vals in output_values.items():
            if method == "average":
                y_s = [list(map(lambda x: sum(x) / len(x), zip(*vals)))]
            elif method == "median":
                y_s = [list(map(lambda x: statistics.median(x), zip(*vals)))]

            data = pd.DataFrame(
                [
                    list(xs + (y,))
                    for xs, y in zip(
                        list(
                            itertools.product(
                                input_values[var][0], input_values[var][1]
                            )
                        ),
                        y_s[0],
                    )
                ]
            )
            data.columns = ["X1", "X2", "Y"]

            # Create new curve
            new_curve = Curve(eqp_type=self.eqp_type, c_type=ref_setofcurves[var].type)

            # Assign curve attributes, assume no min/max
            # TODO: Allow min/max to be passed by user
            new_curve.out_var = var
            new_curve.units = self.sets_of_curves[0].curves[0].units
            new_curve.x_min = -999
            new_curve.y_min = -999
            new_curve.x_max = 999
            new_curve.y_max = 999
            new_curve.out_min = -999
            new_curve.out_max = 999
            if not "normalization" in ranges[var].keys():
                raise ValueError(
                    "Normalization point not provided, the curve cannot be created."
                )
            norm = ranges[var]["normalization"]
            if isinstance(norm, float):
                ref_x = norm
                ref_y = 0
            else:
                ref_x, ref_y = norm
            new_curve.ref_x = ref_x
            new_curve.ref_y = ref_y
            # TODO: update fields below when adding new equipment
            if self.eqp_type == "chiller":
                self.ref_evap_fluid_flow = 0
                self.ref_cond_fluid_flow = 0
                if agg_set_of_curves.model == "ect_lwt":
                    self.ref_lwt = ref_y
                    self.ref_ect = ref_x
                else:
                    raise ValueError("Algorithm not supported.")

            # Find curves coefficients
            new_curve.regression(data)

            # Normalize curve to reference point
            new_curve.normalized(data, ref_x, ref_y)

            agg_set_of_curves.curves.append(new_curve)
        return agg_set_of_curves


class SetofCurves:
    def __init__(self, eqp_type):
        self.name = ""
        self.curves = []
        self.validity = 1.0
        self.source = ""
        self.eqp_type = eqp_type
        self.model = ""
        if eqp_type == "chiller":
            self.ref_cap = 0
            self.ref_cap_unit = ""
            self.ref_eff = 0
            self.ref_eff_unit = ""
            self.comp_type = ""
            self.cond_type = ""
            self.comp_speed = ""
            self.sim_engine = ""
            self.min_plr = 0
            self.min_unloading = 0
            self.max_plr = 0
            if self.cond_type == "water":
                self.plotting_range = {
                    "eir-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 35,
                    },
                    "cap-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 35,
                    },
                    "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": 50},
                    "eir-f-plr-dt": {
                        "x1_min": 0.3,
                        "x1_max": 1,
                        "x1_norm": 1,
                        "nbval": 50,
                        "x2_min": 28.33,
                        "x2_max": 28.33,
                        "x2_norm": 28.33,
                    },
                }
            else:
                self.plotting_range = {
                    "eir-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 29.44,
                    },
                    "cap-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": 29.44,
                    },
                    "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": 50},
                    "eir-f-plr-dt": {
                        "x1_min": 0.3,
                        "x1_max": 1,
                        "x1_norm": 1,
                        "nbval": 50,
                        "x2_min": 22.77,
                        "x2_max": 22.77,
                        "x2_norm": 22.77,
                    },
                }

    def get_data_for_plotting(self, curve, norm):
        var = curve.out_var
        nb_vals = self.plotting_range[var]["nbval"]
        x1_min = self.plotting_range[var]["x1_min"]
        x1_max = self.plotting_range[var]["x1_max"]
        x_1_vals = np.linspace(x1_min, x1_max, nb_vals)

        if "x2_min" in self.plotting_range[var].keys():
            x2_min = self.plotting_range[var]["x2_min"]
            x2_max = self.plotting_range[var]["x2_max"]
            x_2_vals = np.linspace(x2_min, x2_max, nb_vals)
        else:
            x_2_vals = [0]

        y = []
        for v in range(nb_vals):
            if "x2_min" in self.plotting_range[var].keys():
                norm_fac = (
                    curve.evaluate(
                        self.plotting_range[var]["x1_norm"],
                        self.plotting_range[var]["x2_norm"],
                    )
                    if norm
                    else 1
                )
                y_val = curve.evaluate(x_1_vals[v], x_2_vals[v]) / norm_fac
            else:
                norm_fac = (
                    curve.evaluate(
                        self.plotting_range[var]["x1_norm"],
                        self.plotting_range[var]["x1_norm"],
                    )
                    if norm
                    else 1
                )
                y_val = curve.evaluate(x_1_vals[v], x_1_vals[v]) / norm_fac
            y.append(y_val)

        x = x_1_vals if len(set(x_1_vals)) > len(set(x_2_vals)) else x_2_vals

        return [x, y]

    def plot(self, out_var=[], axes=[], norm=True, color="Black", alpha=0.3):
        """Plot set of curves.

        :param list() out_var: List of the output variables to plot, e.g. `eir-f-t`, `eir-f-plr`, `cap-f-t`.
                               Refer to JSON files structure for other output variables
        :param matplotlib.pyplot.axes axes: Matplotlib pyplot axes
        :param boolean norm: Normalize plot to reference values
        :param str color: Set of curves color
        :param float alpha: Transparency of the curves (0-1).
        :return: Plotting success
        :rtype: boolean

        """
        for i, var in enumerate(out_var):
            for curve in self.curves:
                if curve.out_var == var:
                    x, y = self.get_data_for_plotting(curve, norm)
                    axes[i].plot(x, y, color=color, alpha=alpha)
                    axes[i].set_title(var)

        return True

    def export(self, path="./curves", fmt="idf"):
        """Export curves to simulation engine input format.

        :param str path: Path and file name, do not include the extension,
                         it will be added based on the simulation engine
                         of the SetofCurves() object.
        :param str fmt: Input format type, currently not used. TODO: json, idf, inp.
        :return: Success
        :rtype: boolean

        """
        curve_export = ""
        for curve in self.curves:
            curve_type = curve.type
            if self.sim_engine == "energyplus":
                if curve_type == "quad":
                    cuvre_type = "Curve:Quadratic"
                elif curve_type == "bi_quad":
                    cuvre_type = "Curve:Biquadratic"
                elif curve_type == "bi_cub":
                    cuvre_type = "Curve:Bicubic"
                curve_export += (
                    "\n{},\n".format(cuvre_type)
                    if len(curve_export)
                    else "{},\n".format(cuvre_type)
                )
                curve_export += "   {}_{},\n".format(self.name, curve.out_var)
                for i in range(1, curve.nb_coeffs() + 1):
                    curve_export += "   {},\n".format(
                        getattr(curve, "coeff{}".format(i))
                    )
                curve_export += (
                    "   {},\n".format(curve.x_min) if curve.x_min else "    ,\n"
                )
                curve_export += (
                    "   {},\n".format(curve.x_max) if curve.x_max else "    ,\n"
                )
                if curve_type != "quad":
                    curve_export += (
                        "   {},\n".format(curve.y_min) if curve.y_min else "    ,\n"
                    )
                    curve_export += (
                        "   {},\n".format(curve.y_max) if curve.y_max else "    ,\n"
                    )
                curve_export += (
                    "   {},\n".format(curve.out_min) if curve.out_min else "    ,\n"
                )
                curve_export += (
                    "   {};\n".format(curve.out_max) if curve.out_max else "    ;\n"
                )
            else:
                # TODO: implement export to DOE-2 format
                raise ValueError(
                    "Export to the {} input format is not yet implemented.".format(
                        self.sim_engine
                    )
                )
        filen = open(path + self.name + ".{}".format(fmt), "w+")
        filen.write(curve_export)
        return True

    def remove_curve(self, out_var):
        """
        Remove curve for a particular output variables from the set

        :param string out_var: Name of the output variable to remove from the set.

        """
        curves_to_del = []
        for c in self.curves:
            if c.out_var == out_var:
                curves_to_del.append(c)
        for c in self.curves:
            if c in curves_to_del:
                self.curves.remove(c)

    def list_to_dict(self):
        """
        Convert curves from the set from a list to a dictionary, the key being the output variable type.

        :return: Dictionary of curves
        :rtype: dict
        """
        curves = {}
        for c in self.curves:
            curves[c.out_var] = c
        return curves


class Curve:
    def __init__(self, eqp_type, c_type):
        # General charactersitics
        self.out_var = ""
        self.type = c_type
        self.units = "si"
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.out_min = 0
        self.out_max = 0
        self.ref_x = 0
        self.ref_y = 0
        if self.type == "quad":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
        elif self.type == "bi_quad":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
            self.coeff4 = 0
            self.coeff5 = 0
            self.coeff6 = 0
        elif self.type == "bi_cub":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
            self.coeff4 = 0
            self.coeff5 = 0
            self.coeff6 = 0
            self.coeff7 = 0
            self.coeff8 = 0
            self.coeff9 = 0
            self.coeff10 = 0

        # Equipment specific charcatertics
        if eqp_type == "chiller":
            self.ref_evap_fluid_flow = 0
            self.ref_cond_fluid_flow = 0
            self.ref_lwt = 6.67
            self.ref_ect = 29.4
            self.ref_lct = 35

    def evaluate(self, x, y):
        """Return the output of a curve.

        :param float x: First curve independent variable
        :param float y: Second curve independent variable
        :return: Curve output
        :rtype: float

        """
        # Catch nulls
        if self.out_min is None:
            self.out_min = -999
        if self.out_max is None:
            self.out_max = 999
        if self.x_min is None:
            self.x_min = -999
        if self.x_max is None:
            self.x_max = 999
        if self.y_min is None:
            self.y_min = -999
        if self.y_max is None:
            self.y_max = 999

        x = min(max(x, self.x_min), self.x_max)
        y = min(max(y, self.y_min), self.y_max)

        if self.type == "bi_quad":
            out = (
                self.coeff1
                + self.coeff2 * x
                + self.coeff3 * x ** 2
                + self.coeff4 * y
                + self.coeff5 * y ** 2
                + self.coeff6 * x * y
            )
            return min(max(out, self.out_min), self.out_max)
        if self.type == "bi_cub":
            out = (
                self.coeff1
                + self.coeff2 * x
                + self.coeff3 * x ** 2
                + self.coeff4 * y
                + self.coeff5 * y ** 2
                + self.coeff6 * x * y
                + self.coeff7 * x ** 3
                + self.coeff8 * y ** 3
                + self.coeff9 * y * x ** 2
                + self.coeff10 * x * y ** 2
            )
            return min(max(out, self.out_min), self.out_max)
        if self.type == "quad":
            out = self.coeff1 + self.coeff2 * x + self.coeff3 * x ** 2
            return min(max(out, self.out_min), self.out_max)

    def nb_coeffs(self):
        """Find number of curve coefficients.

        :return: Number of curve coefficients
        :rtype: int

        """
        ids = []
        for key in list(self.__dict__.keys()):
            if "coeff" in key:
                ids.append(int(key.split("coeff")[-1]))
        return max(ids)

    def regression(self, data):
        """Find curve coefficient by running a multivariate linear regression.

        :param DataFrame() data: DataFrame() object with the following columns: "X1","X1^2","X2","X2^2","X1*X2", "Y"

        """
        # TODO: implement bi_cubic
        if self.type == "quad":
            data["X1^2"] = data["X1"] * data["X1"]
            X = data[["X1", "X1^2"]]
            y = data["Y"]

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            self.coeff1, self.coeff2, self.coeff3 = model.params
        elif self.type == "bi_quad":
            data["X1^2"] = data["X1"] * data["X1"]
            data["X2^2"] = data["X2"] * data["X2"]
            data["X1*X2"] = data["X1"] * data["X2"]

            X = data[["X1", "X1^2", "X2", "X2^2", "X1*X2"]]
            y = data["Y"]

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            (
                self.coeff1,
                self.coeff2,
                self.coeff3,
                self.coeff4,
                self.coeff5,
                self.coeff6,
            ) = model.params
            r_sqr = model.rsquared
            if r_sqr < 0.8:
                print(
                    "Performance of the regression for {} is poor, r2: {}".format(
                        self.out_var, round(r_sqr, 2)
                    )
                )

    def get_out_reference(self):
        if "-t" in self.out_var:
            x_ref = self.ref_lwt
            y_ref = self.ref_ect
        else:
            x_ref = 1
            y_ref = 0
        return self.evaluate(x_ref, y_ref)

    def normalized(self, data, x_norm, y_norm):
        """Normalize curve around the reference data points.

        :param float x_norm: First independent variable normalization points.
        :param float y_norm: Second independent variable normalization points.
        :param DataFrame() data: DataFrame() object with the following columns: "X1","X1^2","X2","X2^2","X1*X2", "Y".

        """
        data["Y"] = data.apply(
            lambda row: self.evaluate(row["X1"], row["X2"])
            / self.evaluate(x_norm, y_norm),
            axis=1,
        )
        self.regression(data)
