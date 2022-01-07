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
        part_eff_ref_std="ahri_550/590",
        part_eff_alt=0,
        part_eff_unit_alt="",
        part_eff_ref_std_alt="ahri_550/590",
        full_eff_alt=0,
        full_eff_unit_alt="",
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
        self.full_eff_alt = full_eff_alt
        self.full_eff_unit_alt = full_eff_unit_alt
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.part_eff_ref_std = part_eff_ref_std
        self.part_eff_alt = part_eff_alt
        self.part_eff_unit_alt = part_eff_unit_alt
        self.part_eff_ref_std_alt = part_eff_ref_std_alt
        self.min_unloading = min_unloading
        self.model = model
        self.sim_engine = sim_engine
        self.set_of_curves = set_of_curves
        if self.condenser_type == "water":
            if self.part_eff_ref_std == "ahri_550/590":
                lwt = (44.0 - 32.0) * 5 / 9
                ect = (85.0 - 32.0) * 5 / 9
                lct = (94.3 - 32.0) * 5 / 9
            elif self.part_eff_ref_std == "ahri_551/591":
                lwt = 7.0
                ect = 30.0
                lct = 35.0

            if self.model == "ect_lwt":
                self.plotting_range = {
                    "eir-f-t": {
                        "x1_min": lwt,
                        "x1_max": lwt,
                        "x1_norm": lwt,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": ect,
                    },
                    "cap-f-t": {
                        "x1_min": lwt,
                        "x1_max": lwt,
                        "x1_norm": lwt,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 40,
                        "x2_norm": ect,
                    },
                    "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": 50},
                }
            elif self.model == "lct_lwt":
                self.plotting_range = {
                    "eir-f-t": {
                        "x1_min": lwt,
                        "x1_max": lwt,
                        "x1_norm": lwt,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 60,
                        "x2_norm": lct,
                    },
                    "cap-f-t": {
                        "x1_min": lwt,
                        "x1_max": lwt,
                        "x1_norm": lwt,
                        "nbval": 50,
                        "x2_min": 10,
                        "x2_max": 60,
                        "x2_norm": lct,
                    },
                    "eir-f-plr": {
                        "x1_min": lct,
                        "x1_max": lct,
                        "x1_norm": lct,
                        "nbval": 50,
                        "x2_min": 0.0,
                        "x2_max": 1.0,
                        "x2_norm": 1.0,
                    },
                }
            else:
                raise ValueError("Algorithm not supported.")
        elif self.condenser_type == "air":
            if self.part_eff_ref_std == "ahri_550/590":
                lwt = (44.0 - 32.0) * 5 / 9
                ect = (95.0 - 32.0) * 5 / 9
                lct = -999  # does not apply
            elif self.part_eff_ref_std == "ahri_551/591":
                lwt = 7.0
                ect = 35.0
                lct = -999  # does not apply

            self.plotting_range = {
                "eir-f-t": {
                    "x1_min": lwt,
                    "x1_max": lwt,
                    "x1_norm": lwt,
                    "nbval": 50,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "cap-f-t": {
                    "x1_min": lwt,
                    "x1_max": lwt,
                    "x1_norm": lwt,
                    "nbval": 50,
                    "x2_min": 10,
                    "x2_max": 40,
                    "x2_norm": ect,
                },
                "eir-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": 50},
            }

        self.ref_lwt, self.ref_ect, self.ref_lct = lwt, ect, lct

    def get_ref_values(self, out_var):
        if "x2_norm" in list(self.plotting_range[out_var].keys()):
            return [
                self.plotting_range[out_var]["x1_norm"],
                self.plotting_range[out_var]["x2_norm"],
            ]
        else:
            return [self.plotting_range[out_var]["x1_norm"], 0.0]

    def get_ref_cond_flow_rate(self):

        """
        Method to compute the reference flow-rate given ref_cap, full_eff, ref_lct and ref_lwt
        :return:
        """

        # Convert reference capacity if needed
        if self.ref_cap_unit != "kW":
            evap_cap_ton = Units(value=self.ref_cap, unit=self.ref_cap_unit)
            evap_power = evap_cap_ton.conversion(new_unit="kW")
        else:
            evap_power = self.ref_cap

        if self.ref_cap_unit != "kw/ton":
            ref_cap = Units(value=self.ref_cap, unit=self.ref_cap_unit)
            ref_cap = ref_cap.conversion(new_unit="kw/ton")
        else:
            ref_cap = self.ref_cap

        # Convert reference efficiency if needed
        if self.full_eff_unit != "kw/ton":
            full_eff_unit = Units(value=self.full_eff, unit=self.full_eff_unit)
            full_eff = full_eff_unit.conversion(
                new_unit="kw/ton"
            )  # full eff needs to be in kW/ton
        else:
            full_eff = self.full_eff

        # Retrieve curves
        curves = self.get_chiller_curves()
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_plr = curves["eir_f_plr"]

        cap_f_lwt_lct_rated = cap_f_t.evaluate(self.ref_lwt, self.ref_lct)
        cap_f_lwt_lct = cap_f_t.evaluate(self.ref_lwt, self.ref_lct)
        cap_op = 1.0 * cap_f_lwt_lct
        plr = 1.0 * cap_f_lwt_lct_rated / cap_op

        # Calculate compressor power [kW]
        comp_power = (
            ref_cap
            * full_eff
            * cap_f_lwt_lct
            * eir_f_t.evaluate(self.ref_lwt, self.ref_lct)
            * eir_f_plr.evaluate(self.ref_lct, plr)
        )
        cond_cap = evap_power + comp_power

        # Determine the specific heat capacity of water [kJ/kg.K]
        c_p = (
            CP.PropsSI(
                "C",
                "P",
                101325,
                "T",
                0.5 * (self.ref_ect + self.ref_lct) + 273.15,
                "Water",
            )
            / 1000
        )

        # Determine density of water [kg/m3]
        rho = CP.PropsSI(
            "D", "P", 101325, "T", 0.5 * (self.ref_ect + self.ref_lct) + 273.15, "Water"
        )

        # Determine condenser flow rate at reference conditions [m3/s]
        ref_cond_flow_rate = cond_cap / ((self.ref_lct - self.ref_ect) * c_p * rho)

        return ref_cond_flow_rate

    def generate_set_of_curves(
        self,
        method="typical",
        pop_size=100,
        tol=0.0025,
        max_gen=300,
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

    def get_eir_ref(self, alt):
        # Retrieve equipment efficiency and unit
        if alt:
            kwpton_ref = self.full_eff_alt
            kwpton_ref_unit = self.full_eff_unit_alt
        else:
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
        eir_ref = 1 / (ton_to_kbtu / kwpton_ref / kbtu_to_kw)

        return eir_ref

    def calc_eff_ect(self, cap_f_t, eir_f_t, eir_f_plr, eir_ref, ect, lwt, load):
        # Temperature adjustments
        dt = ect - lwt
        cap_f_lwt_ect = cap_f_t.evaluate(lwt, ect)
        eir_f_lwt_ect = eir_f_t.evaluate(lwt, ect)
        cap_op = cap_f_lwt_ect

        # PLR adjustments
        plr = load * cap_f_t.evaluate(lwt, self.ref_ect) / cap_op
        if plr <= self.min_unloading:
            plr = self.min_unloading
        eir_plr = eir_f_plr.evaluate(plr, dt)

        # Efficiency calculation
        eir = eir_ref * eir_f_lwt_ect * eir_plr / plr

        return eir

    def calc_rated_eff(self, eff_type, unit="kw/ton", output_report=False, alt=False):
        """Calculate chiller efficiency.

        :param str eff_type: chiller performance indicator, currently supported `full` (full load rating)
                             and `part` (part load rating)
        :param str unit: Unit of the efficiency indicator
        :return: chiller performance indicator
        :rtype: float

        """

        # Conversion factors
        # TODO: remove these and use the unit class
        ton_to_kbtu = 12
        kbtu_to_kw = 3.412141633

        # Get reference eir
        eir_ref = self.get_eir_ref(alt)
        load_ref = 1

        # Rated conditions as per AHRI Std 551/591
        loads = [1, 0.75, 0.5, 0.25]

        # List of equipment efficiency for each load
        kwpton_lst = []

        # Temperatures at rated conditions
        ect, lwt = self.get_rated_temperatures(alt)

        # Retrieve curves
        curves = self.get_chiller_curves()
        cap_f_t = curves["cap_f_t"]
        eir_f_t = curves["eir_f_t"]
        eir_f_plr = curves["eir_f_plr"]

        try:
            for idx, load in enumerate(
                loads
            ):  # Calculate efficiency for each testing conditions
                if self.model == "ect_lwt":  # DOE-2 chiller model
                    # Efficiency calculation
                    eir = self.calc_eff_ect(
                        cap_f_t, eir_f_t, eir_f_plr, eir_ref, ect[idx], lwt, load
                    )

                elif self.model == "lct_lwt":  # Reformulated EIR chiller model
                    # Determine water properties
                    c_p = (
                        CP.PropsSI("C", "P", 101325, "T", ect[idx] + 273.15, "Water")
                        / 1000
                    )  # kJ/kg.K
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
                            self.set_of_curves[0].ref_cond_fluid_flow * rho,
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
                            self.set_of_curves[0].ref_cond_fluid_flow * rho,
                            c_p,
                        ]

                    # Determine leaving condenser temperature
                    lct = self.get_lct(ect[idx], args)

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

                    if eir < 0:
                        return 999

                else:
                    return -999

                # Convert efficiency to kW/ton
                kwpton = eir / kbtu_to_kw * ton_to_kbtu

                if output_report:
                    cap_ton = self.ref_cap
                    if self.ref_cap_unit != "ton":
                        cap_ton = Units(self.ref_cap, self.ref_cap_unit).conversion(
                            "ton"
                        )
                    part_report = f"""At {str(round(load * 100.0, 0)).replace('.0', '')}% load and AHRI rated conditions:
                    - Entering condenser temperature: {round(ect[idx], 2)},
                    - Leaving chiller temperature: {round(lwt, 2)},
                    - Part load ratio: {round(plr, 2)},
                    - Operating capacity: {round(cap_op * cap_ton, 2)} ton,
                    - Power: {round(kwpton * cap_op * cap_ton, 2)} kW,
                    - Efficiency: {round(kwpton, 3)} kW/ton
                    """
                    print(part_report)

                # Store efficiency for IPLV calculation
                kwpton_lst.append(eir / kbtu_to_kw * ton_to_kbtu)

                # Stop here for full load calculations
                if eff_type == "full" and idx == 0:
                    if unit != "kw/ton":
                        kwpton = Units(kwpton, "kw/ton").conversion(unit)
                    return kwpton

            # Coefficients from AHRI Std 551/591
            iplv = 1 / (
                (0.01 / kwpton_lst[0])
                + (0.42 / kwpton_lst[1])
                + (0.45 / kwpton_lst[2])
                + (0.12 / kwpton_lst[3])
            )

            if output_report:
                print(f"IPLV: {round(iplv, 3)} kW/ton")
        except:
            return -999

        # Convert IPLV to desired unit
        if unit != "kw/ton":
            iplv_org = Units(iplv, "kw/ton")
            iplv = iplv_org.conversion(unit)

        return iplv

    def get_rated_temperatures(self, alt):
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std
        if std == "ahri_551/591":  # IPLV.SI
            lwt = 7.0
            if self.condenser_type == "air":
                ect = [35.0, 27.0, 19.0, 13.0]
            elif self.condenser_type == "water":
                ect = [30.0, 24.5, 19.0, 19.0]
        elif std == "ahri_550/590":  # IPLV.IP
            lwt = 44.0
            if self.condenser_type == "air":
                ect = [95.0, 80.0, 65.0, 55.0]
            elif self.condenser_type == "water":
                ect = [85.0, 75.0, 65.0, 65.0]
            # Convert to SI
            lwt = (lwt - 32.0) * 5 / 9
            ect = [(t - 32.0) * 5 / 9 for t in ect]
        else:
            raise ValueError("Reference standard provided isn't implemented.")
        return [ect, lwt]

    def get_chiller_curves(self):
        curves = {}
        for curve in self.set_of_curves:
            if curve.out_var == "cap-f-t":
                curves["cap_f_t"] = curve
            elif curve.out_var == "eir-f-t":
                curves["eir_f_t"] = curve
            else:
                curves["eir_f_plr"] = curve

        return curves

    def get_lct(self, ect, args):
        result = optimize.root_scalar(
            self.cond_inlet_temp_residual,
            args=(args),
            method="secant",
            x0=ect + 0.1,
            x1=ect + 10,
            rtol=0.001,
        )
        lct = result.root
        return lct

    def cond_inlet_temp_residual(self, lct, args):
        # Get arguments
        lwt, cap_f_t, eir_f_t, eir_f_plr, load, cap_f_lwt_lct_rated, ref_cop, ect, m_c, c_p = (
            args
        )

        # Temperature dependent curve modifiers
        cap_f_lwt_lct = cap_f_t.evaluate(lwt, lct)
        eir_f_lwt_lct = eir_f_t.evaluate(lwt, lct)

        # Convert reference capacity to kW
        if self.ref_cap_unit != "kW":
            ref_cap_org = Units(value=self.ref_cap, unit=self.ref_cap_unit)
            ref_cap = ref_cap_org.conversion(new_unit="kW")
        else:
            ref_cap = self.ref_cap

        # Operating variables
        cap_op = ref_cap * cap_f_lwt_lct
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
