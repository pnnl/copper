"""
curves.py
====================================
The curves module of Copper handles all operations and manipulation related to curves, set of curves, and sets of curves.
"""

import warnings, json, os

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statistics
import itertools
import logging
from copper.units import *


class SetsofCurves:
    def __init__(self, eqp, sets):
        self.eqp = eqp
        self.eqp_type = eqp.type
        self.sets_of_curves = sets

    def get_aggregated_set_of_curves(
        self, method="weighted-average", N=None, ranges={}, misc_attr={}
    ):
        """Determine sets of curves based on aggregation.

        :param str method: Type of aggregation, currently supported: 'average', 'median', 'weighted-average', and 'NN-weighted-average' as in nearest neighbor weighted average.
        :param int N: Number of neighbor used to the aggregation, only used when the method is 'NN-weighted-average'.
        :param dict ranges: Dictionary that defines the ranges of values for each independent variable used to calculate aggregated dependent variable values.
        :param dict misc_attr: Dictionary that provides values for the aggregated set of curves.
        :return: Aggregated set of curves
        :rtype: SetofCurves

        """
        # Check that all curves are/have:
        # - the same output variables
        # - of the same type
        # - defined for the same type of units

        "first entry -> refsetofcurves"
        ref_setofcurves = self.sets_of_curves[0].list_to_dict()
        avail_types = {}
        for set_of_curves in self.sets_of_curves:
            if set(ref_setofcurves.keys()) != set_of_curves.list_to_dict().keys():
                raise ValueError(
                    "The output variables in each set of curves are not consistently the same, aggregated set of curves cannot currently be determined."
                )
            # Retrieve type curve types used for each output variables
            for c in set_of_curves.curves:
                if c.out_var in avail_types.keys():
                    avail_types[c.out_var].append(c.type)
                else:
                    avail_types[c.out_var] = [c.type]
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
                input_values[c.out_var].append(np.linspace(min_val, max_val, 20))
            # Add 0s for second independent variables for univariate curves
            if len(input_values[c.out_var]) == 1:
                input_values[c.out_var].append(np.linspace(0.0, 0.0, 20))

        output_values = {}
        # Calculate values of dependent variables using the user-specified ranges
        for set_of_curves in self.sets_of_curves:
            for c in set_of_curves.curves:
                if not "normalization" in ranges[c.out_var].keys():
                    raise ValueError(
                        "Normalization point not provided, the curve cannot be created."
                    )

                # TODO: move assignement to Curve() class and values ot Chiller() class
                if c.eqp.model == "lct_lwt":
                    if c.out_var == "eir-f-t":
                        c.ref_x = c.ref_lwt
                        c.ref_y = c.ref_lct
                    if c.out_var == "cap-f-t":
                        c.ref_x = c.ref_lwt
                        c.ref_y = c.ref_lct
                    if c.out_var == "eir-f-plr":
                        c.ref_x = c.ref_lct
                        c.ref_y = 1.0  # plr = 1.0
                elif c.eqp.model == "ect_lwt":
                    if c.out_var == "eir-f-t":
                        c.ref_x = c.ref_lwt
                        c.ref_y = c.ref_ect
                    if c.out_var == "cap-f-t":
                        c.ref_x = c.ref_lwt
                        c.ref_y = c.ref_ect
                    if c.out_var == "eir-f-plr":
                        c.ref_x = 1.0  # plr = 1.0
                        c.ref_y = 0.0  # no dependent variable

                norm = ranges[c.out_var]["normalization"]
                if isinstance(norm, float):
                    ref_x = norm
                    ref_y = 0
                else:
                    ref_x, ref_y = norm

                output_value = [
                    c.evaluate(x, y)
                    * c.evaluate(ref_x, ref_y)
                    / c.evaluate(c.ref_x, c.ref_y)
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
        agg_set_of_curves = SetofCurves()
        for att, att_val in misc_attr.items():
            setattr(agg_set_of_curves, att, att_val)

        # Determine aggregated values for dependent variables
        for var, vals in output_values.items():
            if method == "average":
                y_s = [list(map(lambda x: sum(x) / len(x), zip(*vals)))]
            elif method == "median":
                y_s = [list(map(lambda x: statistics.median(x), zip(*vals)))]
            elif method == "weighted-average":
                df, _ = self.nearest_neighbor_sort(target_attr=misc_attr)
                sorted_vals = list(
                    map(vals.__getitem__, df.index.values)
                )  # adding this incase of NaN in dfs
                y_s = [
                    list(
                        map(lambda x: np.dot(df["score"].values, x), zip(*sorted_vals))
                    )
                ]
            elif method == "NN-weighted-average":
                # first make sure that the user has specified to pick N values
                try:
                    assert N is not None
                except AssertionError:
                    logging.critical("Need to specify number of nearest neighbors N")
                df, _ = self.nearest_neighbor_sort(target_attr=misc_attr, N=N)
                sorted_vals = list(map(vals.__getitem__, df.index.values))
                y_s = [
                    list(
                        map(lambda x: np.dot(df["score"].values, x), zip(*sorted_vals))
                    )
                ]

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
            new_curve = Curve(eqp=self.eqp, c_type="")  # curve type is set later on

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
            # TODO: Move following statement to Chiller class
            if self.eqp_type == "chiller":
                if agg_set_of_curves.model == "ect_lwt":
                    self.ref_lwt = ref_y
                    self.ref_ect = ref_x
                elif agg_set_of_curves.model == "lct_lwt":
                    self.ref_lwt = ref_y
                    self.ref_lct = ref_x
                else:
                    raise ValueError("Algorithm not supported.")

            # Find curves coefficients
            new_curve.regression(data, list(set(avail_types[new_curve.out_var])))

            # Normalize curve to reference point
            new_curve.normalized(data, ref_x, ref_y)

            agg_set_of_curves.curves.append(new_curve)

        # Determine reference condenser flow rate
        # TODO: Move following statement to Chiller class
        if self.eqp_type == "chiller":
            self.eqp.set_of_curves = agg_set_of_curves.curves
            if self.eqp.condenser_type == "water":
                cond_flow_rate = self.eqp.get_ref_cond_flow_rate()
                for c in agg_set_of_curves.curves:
                    c.ref_cond_fluid_flow = cond_flow_rate
                    c.ref_evap_fluid_flow = 0

        return agg_set_of_curves

    def nearest_neighbor_sort(
        self, target_attr=None, vars=["ref_cap", "full_eff"], N=None
    ):
        """This function performs the weighted average and the nearest neighbor approach.

        :param dict target_attr: Target attributes we want to match
        :param list vars: The variables we want to use to compute our l2 score. note COP will be added
        :param int N: Indicates the number of nearest neighbors to consider. N=None for weighted-average
        :param pandas.DataFrame df: Pandas dataframe with selected chiller names and the associated weightings
        :return: Index of set_of_curve that should be the closest fit
        :rtype: int

        """

        if target_attr is None:
            raise ValueError("target_attr cannot be None. Enter valid attributes")

        vars_not_in_dict = self.check_vars_in_dictionary(
            vars=vars, target_attr=target_attr
        )

        try:
            assert not vars_not_in_dict
        except AssertionError as err:
            err.args = (
                "The following variables not in equipment library: ",
                vars_not_in_dict,
            )
            raise

        df_list = []
        data = {}

        if target_attr is None:
            logging.error("Enter valid attributes. Returning Empty DataFrame")
            df = pd.DataFrame
            best_idx = None
        else:
            for setofcurve in self.sets_of_curves:
                data["name"] = [setofcurve.name]
                for var in vars:
                    data[var] = [setofcurve.eqp.__dict__[var]]
                df_list.append(pd.DataFrame(data))
            df = pd.concat(df_list)

            # check if there is only a single curve.
            # in that case, we select that curve, even if there ar NaN values
            if len(df) == 1 and len(df.dropna()) == 0:
                for var in vars:
                    df[var] = target_attr[var]
            else:
                df = df.dropna()
            assert len(df) > 0

            if N is not None:
                df, target_attr, best_idx = self.normalize_vars(
                    df=df, target_attr=target_attr, N=N, vars=vars
                )
            else:
                df, target_attr, best_idx = self.normalize_vars(
                    df=df, target_attr=target_attr, vars=vars
                )

        return df, best_idx

    def normalize_vars(
        self,
        df,
        target_attr=None,
        vars=["ref_cap", "full_eff"],
        epsilon=0.00001,
        weights=None,
        N=None,
    ):
        """Normalize curve outputs.

        :param pandas.DataFrame df: Input dataframe containing the variable inputs
        :param dict target_attr: Reference targets with respect to which l2 score needs to computed
        :param list vars: List of strings for variables we want to normalize
        :param list weights: Weights associated with each variable in vars
        :param int N: Number of nearest neighbors. It should be none unless method is 'NN-weighted-average'
        :return: Dataframe with added columns with normalized variables, dict with added normalized values of var in vars, index of the best curve
        :rtype: list

        """

        if weights is None:
            weights = [(1.0) / len(vars) for var in vars]

        for var in vars:
            var_name = var + "_norm"
            if len(df) == 1:
                df[var_name] = 1
            else:
                df[var_name] = (df[var] - df[var].mean()) / (df[var].std() + epsilon)

            if target_attr is not None and var in target_attr.keys():
                target_attr[var_name] = (target_attr[var] - df[var].mean()) / (
                    df[var].std() + epsilon
                )
            else:
                logging.error(
                    "Please enter valid target_attr. Also the variable name must be in dictionary"
                )
                target_attr[var_name] = None

        # compute the l2 norm
        x = -self.l2_norm(df=df, target_attr=target_attr, weights=weights, vars=vars)

        if len(df) == 1:
            df["score"] = 1
        elif N is not None:
            df["score"] = x
            # first sort and pick top N candidate
            df = df.reset_index(drop=True)
            df = df.sort_values(by="score", ascending=False)
            df = df.iloc[:N]
            df["score"] = self.softmax(df["score"])
        else:
            df["score"] = self.softmax(x)
            df = df.reset_index(drop=True)

        best_curve_idx = df.index.values[np.argmax(df["score"].values)]

        return df, target_attr, best_curve_idx

    def l2_norm(self, df, target_attr, weights, vars=["full_eff", "ref_cap"]):
        """Perform L2 normalization.

        :param pandas.DataFrame df: Dataframe containing the attributes of different equipments for a given equipment type
        :param dict target_attr: Target equipment attribute
        :param list weights: List of weights, must have the same dimensions as vars
        :param list vars: List of string containing variable names to compute the L2 normalization
        :return: L2 scores (same size as df)
        :rtype: numpy.array

        """

        norm_vars = [var + "_norm" for var in vars]
        ref_data = [target_attr[var] for var in norm_vars]
        data = df[norm_vars].values
        Y = np.matmul(np.sqrt((data - ref_data) ** 2), weights)

        return Y

    def softmax(self, x):
        """Softmax function.

        :param float x: Convert distances to scores/weights using the softmax function
        :return: Result of the softmax function
        :rtype: float

        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def check_vars_in_dictionary(self, vars, target_attr):
        """Function to check that the vars specified by the user exists in 'target_attr'.

        :param list vars: Variable to calculate weights with, as specified by the user
        :param list target_attr: Target attributes
        :return: List of variables not in 'target_attr'
        :rtype: list

        """
        not_in_dict = []
        for var in vars:
            if var not in target_attr.keys():
                not_in_dict.append(var)

        return not_in_dict


class SetofCurves:
    def __init__(self):
        self.name = ""
        self.curves = []
        self.eqp = ""

    def get_data_for_plotting(self, curve, norm):
        """Retrieve equipment specific data for plotting set of curves.

        :param Curve curve: Copper curve object
        :param bool norm: Normalize data used for plotting the curves
        :return: Data to be used for plotting the curves
        :rtype: list

        """
        var = curve.out_var
        nb_vals = self.eqp.plotting_range[var]["nbval"]
        x1_min = self.eqp.plotting_range[var]["x1_min"]
        x1_max = self.eqp.plotting_range[var]["x1_max"]
        x_1_vals = np.linspace(x1_min, x1_max, nb_vals)

        if "x2_min" in self.eqp.plotting_range[var].keys():
            x2_min = self.eqp.plotting_range[var]["x2_min"]
            x2_max = self.eqp.plotting_range[var]["x2_max"]
            x_2_vals = np.linspace(x2_min, x2_max, nb_vals)
        else:
            x_2_vals = [0]

        y = []
        for v in range(nb_vals):
            if "x2_min" in self.eqp.plotting_range[var].keys():
                norm_fac = (
                    curve.evaluate(
                        self.eqp.plotting_range[var]["x1_norm"],
                        self.eqp.plotting_range[var]["x2_norm"],
                    )
                    if norm
                    else 1
                )
                y_val = curve.evaluate(x_1_vals[v], x_2_vals[v]) / norm_fac
            else:
                norm_fac = (
                    curve.evaluate(
                        self.eqp.plotting_range[var]["x1_norm"],
                        self.eqp.plotting_range[var]["x1_norm"],
                    )
                    if norm
                    else 1
                )
                y_val = curve.evaluate(x_1_vals[v], x_1_vals[v]) / norm_fac
            y.append(y_val)

        x = x_1_vals if len(set(x_1_vals)) > len(set(x_2_vals)) else x_2_vals

        return [x, y]

    def plot(self, out_var=[], axes=[], norm=True, color="Black", alpha=0.3):
        """Plot a set of curves.

        :param list out_var: List of the output variables to plot, e.g. `eir-f-t`, `eir-f-plr`, `cap-f-t`.
                               Refer to JSON files structure for other output variables
        :param matplotlib.pyplot.axes axes: Matplotlib pyplot axes
        :param bool norm: Normalize plot to reference values
        :param str color: Set of curves color
        :param float alpha: Transparency of the curves (0-1).
        :return: Plotting success
        :rtype: bool

        """
        for i, var in enumerate(out_var):
            for curve in self.curves:
                if curve.out_var == var:
                    x, y = self.get_data_for_plotting(curve, norm)
                    axes[i].plot(x, y, color=color, alpha=alpha)
                    axes[i].set_title(var)

        return True

    def export(self, path="./", fmt="idf", name=""):
        """Export curves to simulation engine input format.

        :param str path: Path and file name, do not include the extension,
                         it will be added based on the simulation engine
                         of the SetofCurves object.
        :param str fmt: Input format type, currently not used. TODO: json, idf, inp.
        :return: Success
        :rtype: bool

        """
        if fmt == "json":
            curve_export = {}
            curve_export[name] = []
        else:
            curve_export = ""
        for curve in self.curves:
            curve_type = curve.type
            self.name = self.name.replace("/", "_").replace(" ", "_")
            if fmt == "idf":
                if curve_type == "quad":
                    curve_type = "Curve:Quadratic"
                elif curve_type == "bi_quad":
                    curve_type = "Curve:Biquadratic"
                elif curve_type == "bi_cub":
                    curve_type = "Curve:Bicubic"
                elif curve_type == "cubic":
                    curve_type = "Curve:Cubic"
                curve_export += (
                    "\n{},\n".format(curve_type)
                    if len(curve_export)
                    else "{},\n".format(curve_type)
                )
                curve_export += "   {}_{},\n".format(self.name, curve.out_var)
                for i in range(1, curve.nb_coeffs() + 1):
                    curve_export += "   {},\n".format(
                        getattr(curve, "coeff{}".format(i))
                    )
                curve_export += (
                    "   {},\n".format(curve.x_min)
                    if curve.x_min
                    else "   0.0,\n"  # TODO: Temporary fix
                )
                curve_export += (
                    "   {},\n".format(curve.x_max) if curve.x_max else "    ,\n"
                )
                if curve_type != "quad" and curve_type != "cubic":
                    curve_export += (
                        "   {},\n".format(curve.y_min) if curve.y_min else "    ,\n"
                    )
                    curve_export += (
                        "   {},\n".format(curve.y_max) if curve.y_max else "    ,\n"
                    )
                curve_export += "   {},\n".format(0) if curve.out_min else "    ,\n"
                curve_export += (
                    "   {};\n".format(curve.out_max) if curve.out_max else "    ;\n"
                )
                filen = open(path + "/" + self.name + ".{}".format(fmt), "w+")
                filen.write(curve_export)
            elif fmt == "csv":
                curve_export += f"{self.name},{curve.out_var},{curve.units},{curve.type},{curve.x_min},{curve.x_max},{curve.y_min},{curve.y_max}"
                for i in range(1, curve.nb_coeffs() + 1):
                    curve_export += ",{}".format(getattr(curve, "coeff{}".format(i)))
                curve_export += "\n"
                if name == "":
                    filen = open(path + "/" + self.name + ".{}".format(fmt), "a+")
                else:
                    filen = open(path + "/" + name + ".{}".format(fmt), "a+")
                filen.write(curve_export)
                curve_export = ""
            elif fmt == "json":
                c = curve.__dict__
                c.pop("eqp")
                curve_export[name].append(c)
        if fmt == "json":
            with open(os.path.join(path, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(curve_export, f, indent=4)

        return True

    def remove_curve(self, out_var):
        """Remove curve for a particular output variable from the set.

        :param str out_var: Name of the output variable to remove from the set

        """
        curves_to_del = []
        for c in self.curves:
            if c.out_var == out_var:
                curves_to_del.append(c)
        for c in self.curves:
            if c in curves_to_del:
                self.curves.remove(c)

    def list_to_dict(self):
        """Convert curves from the set from a list to a dictionary (the key being the output variable type).

        :return: Dictionary of Copper curve objects
        :rtype: dict
        """
        curves = {}
        for c in self.curves:
            curves[c.out_var] = c
        return curves


class Curve:
    def __init__(self, eqp, c_type):
        # General charactersitics
        self.eqp = eqp
        self.out_var = ""
        self.type = c_type
        self.units = "si"
        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None
        self.out_min = None
        self.out_max = None
        self.ref_x = 0
        self.ref_y = 0
        if self.type == "quad":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
        elif self.type == "cubic":
            self.coeff1 = 0
            self.coeff2 = 0
            self.coeff3 = 0
            self.coeff4 = 0
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

        # Equipment specific charactertics
        # TODO: move under a function in the Chiller class
        if self.eqp.type == "chiller":
            self.ref_evap_fluid_flow = 0
            self.ref_cond_fluid_flow = 0
            if self.eqp.part_eff_ref_std == "ahri_550/590":
                self.ref_lwt = (44.0 - 32.0) * 5 / 9
                if self.eqp.condenser_type == "water":
                    self.ref_ect = (85.0 - 32.0) * 5 / 9
                    self.ref_lct = (94.3 - 32.0) * 5 / 9
                else:
                    self.ref_ect = (95.0 - 32.0) * 5 / 9
            elif self.eqp.part_eff_ref_std == "ahri_551/591":
                self.ref_lwt = 7.0
                if self.eqp.condenser_type == "water":
                    self.ref_ect = 30.0
                    self.ref_lct = 35
                else:
                    self.ref_ect = 35.0

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
                + self.coeff3 * x**2
                + self.coeff4 * y
                + self.coeff5 * y**2
                + self.coeff6 * x * y
            )
            return min(max(out, self.out_min), self.out_max)
        if self.type == "bi_cub":
            out = (
                self.coeff1
                + self.coeff2 * x
                + self.coeff3 * x**2
                + self.coeff4 * y
                + self.coeff5 * y**2
                + self.coeff6 * x * y
                + self.coeff7 * x**3
                + self.coeff8 * y**3
                + self.coeff9 * y * x**2
                + self.coeff10 * x * y**2
            )
            return min(max(out, self.out_min), self.out_max)
        if self.type == "quad":
            out = self.coeff1 + self.coeff2 * x + self.coeff3 * x**2
            return min(max(out, self.out_min), self.out_max)
        if self.type == "cubic":
            out = (
                self.coeff1
                + self.coeff2 * x
                + self.coeff3 * x**2
                + self.coeff4 * x**3
            )
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

    def compute_grad(self, x, y, sign_val, threshold=1e-5):
        """Check, for a single curve, if the gradient has the sign we expect. called by check_gradients.

        :return: Verification result
        :rtype: bool

        """
        grad = np.around(
            np.gradient(y, x), 2
        )  # add a small number to get rid of very small negative values
        grad[
            np.abs(grad) <= threshold
        ] = 0  # making sure that small gradients are set to zero to avoid
        sign = np.sign(grad)

        if np.all(np.asarray(y) == 0):  # all values are false
            return False
        elif np.all(
            sign != -sign_val
        ):  # include 0 and +1/-1 gradients. but not gradients of the opposite sign
            return True
        else:
            return False

    def regression(self, data, curve_types):
        """Find curve coefficient by running a multivariate linear regression.

        :param pandas.DataFrame data: Dataframe object with the following columns: 'X1', 'X1^2', 'X2', 'X2^2', 'X1*X2', 'Y'
        :param list curve_types: List of Copper curve types

        """
        # Global R^2
        r_sqr = 0

        # Define expected gradient sign
        if self.out_var == "eir-f-t" or self.out_var == "eir-f-plr":
            sign_val = +1
        elif self.out_var == "cap-f-t":
            sign_val = -1

        # Drop duplicate entries
        data.drop_duplicates(inplace=True)

        # Find model that fits data the best
        if "quad" in curve_types:
            # Prepare data for model
            data["X1^2"] = data["X1"] * data["X1"]
            X = data[["X1", "X1^2"]]
            y = data["Y"]

            # OLS regression
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            reg_r_sqr = model.rsquared

            if reg_r_sqr > r_sqr:
                self.coeff1, self.coeff2, self.coeff3 = model.params
                self.type = "quad"
                r_sqr = reg_r_sqr

        if "cubic" in curve_types:
            # Prepare data for model
            data["X1^2"] = data["X1"] * data["X1"]
            data["X1^3"] = data["X1"] * data["X1"] * data["X1"]
            X = data[["X1", "X1^2", "X1^3"]]
            y = data["Y"]

            # OLS regression
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            reg_r_sqr = model.rsquared

            # Compute independent variable using model
            # to see if curve is monotonic
            vals = []
            c = Curve(eqp=self.eqp, c_type="cubic")
            c.coeff1, c.coeff2, c.coeff3, c.coeff4 = model.params
            for x in data["X1"]:
                vals.append(c.evaluate(x, 0))

            if reg_r_sqr > r_sqr and self.compute_grad(data["X1"], vals, sign_val):
                self.coeff1, self.coeff2, self.coeff3, self.coeff4 = model.params
                self.type = "cubic"
                r_sqr = reg_r_sqr

        if "bi_quad" in curve_types:
            # Prepare data for model
            data["X1^2"] = data["X1"] * data["X1"]
            data["X2^2"] = data["X2"] * data["X2"]
            data["X1*X2"] = data["X1"] * data["X2"]
            X = data[["X1", "X1^2", "X2", "X2^2", "X1*X2"]]
            y = data["Y"]

            # OLS regression
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            reg_r_sqr = model.rsquared
            if reg_r_sqr < 0.8:
                logging.warning(
                    "Performance of the regression for {} is poor, r2: {}".format(
                        self.out_var, round(r_sqr, 2)
                    )
                )
            if reg_r_sqr > r_sqr:
                (
                    self.coeff1,
                    self.coeff2,
                    self.coeff3,
                    self.coeff4,
                    self.coeff5,
                    self.coeff6,
                ) = model.params
                self.type = "bi_quad"
                r_sqr = reg_r_sqr
        if "bi_cub" in curve_types:
            # Prepare data for model
            data["X1^2"] = data["X1"] * data["X1"]
            data["X1^3"] = data["X1"] * data["X1"] * data["X1"]
            data["X2^2"] = data["X2"] * data["X2"]
            data["X2^3"] = data["X2"] * data["X2"] * data["X2"]
            data["X1*X2"] = data["X1"] * data["X2"]
            data["X1^2*X2"] = data["X1^2"] * data["X2"]
            data["X1*X2^2"] = data["X1"] * data["X2^2"]

            X = data[
                [
                    "X1",
                    "X1^2",
                    "X2",
                    "X2^2",
                    "X1*X2",
                    "X1^3",
                    "X2^3",
                    "X1^2*X2",
                    "X1*X2^2",
                ]
            ]
            y = data["Y"]

            # OLS regression
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            reg_r_sqr = model.rsquared
            if reg_r_sqr < 0.8:
                logging.warning(
                    "Performance of the regression for {} is poor, r2: {}".format(
                        self.out_var, round(r_sqr, 2)
                    )
                )
            if reg_r_sqr > r_sqr:
                (
                    self.coeff1,
                    self.coeff2,
                    self.coeff3,
                    self.coeff4,
                    self.coeff5,
                    self.coeff6,
                    self.coeff7,
                    self.coeff8,
                    self.coeff9,
                    self.coeff10,
                ) = model.params
                self.type = "bi_cub"
                r_sqr = reg_r_sqr

    def get_out_reference(self, eqp):
        """Return the reference output of a curve.

        :param eqp: Equipment
        :return: Curve output at reference conditions
        :rtype: float

        """
        x_ref, y_ref = eqp.get_ref_values(self.out_var)
        return self.evaluate(x_ref, y_ref)

    def normalized(self, data, x_norm, y_norm):
        """Normalize curve around the reference data points.

        :param pandas.DataFrame data: Dataframe object with the following columns: 'X1', 'X1^2', 'X2', 'X2^2', 'X1*X2', 'Y'
        :param float x_norm: First independent variable normalization points
        :param float y_norm: Second independent variable normalization points

        """
        # Normalization point
        norm_out = self.evaluate(x_norm, y_norm)
        data["Y"] = data.apply(
            lambda row: self.evaluate(row["X1"], row["X2"]) / norm_out, axis=1
        )

        self.regression(data, [self.type])

    def convert_coefficients_to_ip(self):
        """Convert curve coefficient to imperial units"""
        if self.units == "si":
            data_pts = []
            if "f-t" in self.out_var and "bi" in self.type:
                for x in np.linspace(self.x_min, self.x_max, 5):
                    for y in np.linspace(self.y_min, self.y_max, 5):
                        x_ip = Units(x, "degC")
                        y_ip = Units(y, "degC")
                        data_pt = [
                            x_ip.conversion("degF"),
                            y_ip.conversion("degF"),
                            self.evaluate(x, y),
                        ]
                        data_pts.append(data_pt)
            if len(data_pts) > 0:
                data = pd.DataFrame(data_pts, columns=["X1", "X2", "Y"])
                self.regression(data, self.type)
                self.x_min = Units(self.x_min, "degC").conversion("degF")
                self.x_max = Units(self.x_max, "degC").conversion("degF")
                self.y_min = Units(self.y_min, "degC").conversion("degF")
                self.y_max = Units(self.y_max, "degC").conversion("degF")
                self.units = "ip"
