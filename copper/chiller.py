import CoolProp.CoolProp as CP
from scipy import optimize
from copper.ga import *
from copper.units import *
from copper.curves import *


class chiller:
    def __init__(
        self,
        ref_cap,
        ref_cap_unit,
        full_eff,
        full_eff_unit,
        compressor_type,
        condenser_type,
        compressor_speed,
        part_eff=0,
        part_eff_unit="",
        part_eff_ref_std="ahri_551/591",
        set_of_curves="",
        model="ect_lwt",
        sim_engine="energyplus",
        min_unloading=0.1,
    ):
        self.type = "chiller"
        self.compressor_type = compressor_type
        self.condenser_type = condenser_type
        self.compressor_speed = compressor_speed
        self.ref_cap = ref_cap
        self.ref_cap_unit = ref_cap_unit
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.part_eff_ref_std = part_eff_ref_std
        self.min_unloading = min_unloading
        self.model = model
        self.sim_engine = sim_engine
        self.set_of_curves = set_of_curves
        if self.condenser_type == "water":
            if self.model == "ect_lwt":
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
                        "x1_min": 0,
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
                        "x2_max": 60,
                        "x2_norm": 35,
                    },
                    "cap-f-t": {
                        "x1_min": 6.67,
                        "x1_max": 6.67,
                        "x1_norm": 6.67,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 60,
                        "x2_norm": 35,
                    },
                    "eir-f-plr-dt": {
                        "x1_min": 35.0,
                        "x1_max": 35.0,
                        "x1_norm": 35.0,
                        "nbval": 50,
                        "x2_min": 0.0,
                        "x2_max": 1.0,
                        "x2_norm": 1.0,
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

    def generate_set_of_curves(
        self,
        method="typical",
        pop_size=100,
        tol=0.005,
        max_gen=15000,
        vars="",
        sFac=0.5,
        retain=0.2,
        random_select=0.1,
        mutate=0.95,
        bounds=(6, 10),
        base_curves=[],
    ):
        """Generate a set of curves for a particular chiller() object.

        :param str method: Method used to generate the set of curves, either `typical` or `best_match`

                           - `typical` uses typical curves and modify them to reach a particular IPLV
                           - `best_match` uses curves that best match the chiller object description
        :param int pop_size: Population size used by the genetic algorithm
        :param float tol: Tolerance used by the genetic algorithm to determine if the proposed solution is acceptable
                          The lower, the more stringent
        :param int max_gen: Maximum number of generation
        :param list() vars: List of variable to run the alorithm on
        :param float sFac: Linear fitness normalization factor, the higher the more aggressive the normalization will be
        :param float retain: Probability of retaining an individual in the next generation
        :param float random_select: Probability of randomly selecting an individual to be part of the next generation
        :param float mutate: Probability of an individual to be mutated in the next generation
        :param tuple() bounds: Random modification bounds (TODO: add more details)
        :return: Set of curves object generated by the genetic algorithm that matches the chiller() definition
        :rtype: SetofCurves()

        """
        ga = GA(
            self,
            method,
            pop_size,
            tol,
            max_gen,
            vars,
            sFac,
            retain,
            random_select,
            mutate,
            bounds,
            base_curves,
        )
        return ga.generate_set_of_curves()

    def calc_eff(self, eff_type, unit="kw/ton"):
        """Calculate chiller efficiency.

        :param str eff_type: chiller performance indicator, currently supported `full` (full load rating)
                             and `part` (part load rating)
        :param str unit: Unit of the efficiency indicator
        :return: chiller performance indicator
        :rtype: float

        """
        # Retrieve equipment efficiency and unit
        kwpton_ref = self.full_eff
        kwpton_ref_unit = self.full_eff_unit

        # Convert to kWpton if necessary
        if self.full_eff_unit != "kw/ton":
            kwpton_ref_unit = Units(kwpton_ref, kwpton_ref_unit)
            kwpton_ref = kwpton_ref_unit.conversion("kw/ton")

        # Conversion factors
        # TODO: remove these and use the unit class
        ton_to_kbtu = 12
        kbtu_to_kw = 3.412141633

        # Full load conditions
        load_ref = 1
        eir_ref = 1 / (ton_to_kbtu / kwpton_ref / kbtu_to_kw)

        # Rated conditions as per AHRI Std 551/591
        loads = [1, 0.75, 0.5, 0.25]

        # List of equipment efficiency for each load
        kwpton_lst = []

        # Temperatures at rated conditions
        if self.part_eff_ref_std == "ahri_551/591":  # IPLV.SI
            if self.condenser_type == "air":
                lwt = 7.0
                ect = [35, 27, 19, 13]
            elif self.condenser_type == "water":
                lwt = 7.0
                ect = [30.0, 24.5, 19.0, 19.0]
        elif self.part_eff_ref_std == "ahri_550/590":  # IPLV.IP
            if self.condenser_type == "air":
                lwt = 44.0
                ect = [95.0, 80.0, 65.0, 55.0]
            elif self.condenser_type == "water":
                lwt = 44.0
                ect = [85.0, 75.0, 65.0, 65.0]
            # Convert to SI
            lwt = (lwt - 32.0) * 5 / 9
            ect = [(t - 32.0) * 5 / 9 for t in ect]
        else:
            raise ValueError("Reference standard provided isn't implemented.")

        # Retrieve curves
        for curve in self.set_of_curves:
            if curve.out_var == "cap-f-t":
                cap_f_t = curve
            elif curve.out_var == "eir-f-t":
                eir_f_t = curve
            else:
                eir_f_plr = curve

        try:
            for idx, load in enumerate(
                loads
            ):  # Calculate efficiency for each testing conditions
                if self.model == "ect_lwt":  # DOE-2 chiller model
                    # Temperature adjustments
                    dt = ect[idx] - lwt
                    cap_f_lwt_ect = cap_f_t.evaluate(lwt, ect[idx])
                    eir_f_lwt_ect = eir_f_t.evaluate(lwt, ect[idx])
                    cap_op = load_ref * cap_f_lwt_ect

                    # PLR adjustments
                    plr = load * cap_f_t.evaluate(lwt, ect[0]) / cap_op
                    if plr <= self.min_unloading:
                        plr = self.min_unloading
                    eir_plr = eir_f_plr.evaluate(plr, dt)

                    # Efficiency calculation
                    eir = eir_ref * eir_f_lwt_ect * eir_plr / plr

                elif self.model == "lct_lwt":  # Reformulated EIR chiller model
                    # Determine water properties
                    c_p = CP.PropsSI("C", "P", 101325, "T", ect[idx] + 273.15, "Water")
                    rho = CP.PropsSI("D", "P", 101325, "T", ect[idx] + 273.15, "Water")

                    # Gather arguments for determination fo leaving condenser temperature through iteration
                    if idx == 0:  # Full load rated conditions
                        args = [
                            lwt,
                            cap_f_t,
                            eir_f_t,
                            eir_f_plr,
                            load,
                            -999,
                            1 / eir_ref,
                            ect[idx],
                            self.set_of_curves[0].ref_evap_fluid_flow * rho,
                            c_p,
                        ]
                    else:
                        args = [
                            lwt,
                            cap_f_t,
                            eir_f_t,
                            eir_f_plr,
                            load,
                            cap_f_lwt_lct_rated,
                            1 / eir_ref,
                            ect[idx],
                            self.set_of_curves[0].ref_evap_fluid_flow * rho,
                            c_p,
                        ]

                    # Determine leaving condenser temperature
                    result = optimize.root_scalar(
                        self.cond_inlet_temp_residual,
                        args=(args),
                        method="secant",
                        x0=ect[idx] + 0.1,
                        x1=ect[idx] + 10,
                        rtol=0.001,
                    )
                    lct = result.root

                    # Determine rated capacity curve modifier
                    if idx == 0:
                        cap_f_lwt_lct_rated = cap_f_t.evaluate(lwt, lct)

                    # Temperature adjustments
                    dt = ect[idx] - lwt
                    cap_f_lwt_lct = cap_f_t.evaluate(lwt, lct)
                    eir_f_lwt_lct = eir_f_t.evaluate(lwt, lct)
                    cap_op = load_ref * cap_f_lwt_lct

                    # PLR adjustments
                    plr = load * cap_f_lwt_lct_rated / cap_op
                    if plr <= self.min_unloading:
                        plr = self.min_unloading
                    eir_plr_lct = eir_f_plr.evaluate(lct, plr)

                    # Efficiency calculation
                    eir = eir_ref * eir_f_lwt_lct * eir_plr_lct / plr

                else:
                    return -999

                # Convert efficiency to kW/ton
                kwpton = eir / kbtu_to_kw * ton_to_kbtu

                # Store efficiency for IPLV calculation
                kwpton_lst.append(eir / kbtu_to_kw * ton_to_kbtu)

                # Stop here for full load calculations
                if eff_type == "full" and idx == 0:
                    return kwpton

            # Coefficients from AHRI Std 551/591
            iplv = 1 / (
                (0.01 / kwpton_lst[0])
                + (0.42 / kwpton_lst[1])
                + (0.45 / kwpton_lst[2])
                + (0.12 / kwpton_lst[3])
            )
        except:
            return -999

        # Convert IPLV to desired unit
        if unit != "kw/ton":
            iplv_org = Units(iplv, "kw/ton")
            iplv = iplv_org.conversion(unit)

        return iplv

    def cond_inlet_temp_residual(self, lct, args):
        # Get arguments
        lwt, cap_f_t, eir_f_t, eir_f_plr, load, cap_f_lwt_lct_rated, ref_cop, ect, m_c, c_p = (
            args
        )

        # Temperature dependent curve modifiers
        cap_f_lwt_lct = cap_f_t.evaluate(lwt, lct)
        eir_f_lwt_lct = eir_f_t.evaluate(lwt, lct)

        # Operating varibles
        cap_op = self.ref_cap * cap_f_lwt_lct
        if cap_f_lwt_lct_rated == -999:
            plr = load
        else:
            plr = load * cap_f_lwt_lct_rated / cap_f_lwt_lct

        # PLR and temperature curve modifier
        eir_f_plr_lct = eir_f_plr.evaluate(lct, plr)

        # Evaporator operating capacity
        q_e = cap_op * plr

        # Chiller power
        p = (cap_op / ref_cop) * eir_f_lwt_lct * eir_f_plr_lct

        # Condenser heat transfer
        q_c = p + q_e

        # Store original ECT
        ect_org = ect

        # Recalculate ECT
        ect = lct - q_c / (m_c * c_p)

        return (ect_org - ect) / ect_org
