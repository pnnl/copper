"""
unitary_dx.py
====================================
This is the unitary direct expansion (DX) HVAC equipment module of Copper. The module handles all calculations and data manipulation related to unitary DX equipment.
"""

import CoolProp.CoolProp as CP
from copper.generator import *
from copper.units import *
from copper.curves import *
from copper.library import *
from copper.equipment import *
import logging, json

location = os.path.dirname(os.path.realpath(__file__))
unitary_dx_lib = os.path.join(location, "data", "unitarydirectexpansion_curves.json")
equipment_references = json.load(
    open(os.path.join(location, "data", "equipment_references.json"), "r")
)
log_fan = False


class UnitaryDirectExpansion(Equipment):
    def __init__(
        self,
        full_eff,
        full_eff_unit,
        compressor_type,
        compressor_speed="constant",
        ref_cap_unit="W",
        indoor_fan_power=None,
        part_eff=0,
        ref_gross_cap=None,
        ref_net_cap=None,
        part_eff_unit="",
        set_of_curves=[],
        part_eff_ref_std="ahri_340/360",
        part_eff_ref_std_alt=None,
        model="simplified_bf",
        sim_engine="energyplus",
        condenser_type="air",
        degradation_coefficient=0.115170535550221,  # PLF = 1 - (1 - PLR) * C_D = (1 - C_D) + C_D * PLR Equation 11.63 in AHRI 240/210 (2024)
        indoor_fan_speeds_mapping={
            "1": {
                "fan_flow_fraction": 0.66,
                "fan_power_fraction": 0.4,
                "compressor_stage": 1,
            },
            "2": {
                "fan_flow_fraction": 1.0,
                "fan_power_fraction": 1.0,
                "compressor_stage": 2,
            },
        },
        indoor_fan_curve_coef={
            "type": "cubic",
            "1": 0.040816,
            "2": 0.088035,
            "3": -0.07293,
            "4": 0.944078,
        },
        indoor_fan_speeds=1,
        indoor_fan_curve=False,
        indoor_fan_power_unit="kW",
        compressor_stages=[],
        control_power={},
        control_power_unit="kW",
    ):
        global log_fan
        self.type = "UnitaryDirectExpansion"

        # Input validation and populate default assumptions
        if model != "simplified_bf":
            logging.error("Model must be 'simplified_bf'")
            raise ValueError("Model must be 'simplified_bf'")
        if ref_gross_cap == None:
            if ref_net_cap == None:
                logging.error("Input must be one and only one capacity input")
                raise ValueError("Input must be one and only one capacity input")
            else:
                if indoor_fan_power == None:
                    # This is 400 cfm/ton and 0.365 W/cfm. Equation 11.1 from AHRI 210/240 (2024).
                    indoor_fan_power_unit = "kW"
                    indoor_fan_power_unit = "kW"
                    indoor_fan_power = Units(
                        value=Units(value=ref_net_cap, unit=ref_cap_unit).conversion(
                            new_unit="ton"
                        )
                        * 400
                        * 0.365,
                        unit="W",
                    ).conversion(new_unit=indoor_fan_power_unit)
                    if not log_fan:
                        logging.info(
                            f"Default fan power is based on 400 cfm/ton and 0.365 kW/cfm"
                        )
                        logging.info(
                            f"Default fan power is based on 400 cfm/ton and 0.365 kW/cfm"
                        )
                        log_fan = True
                ref_gross_cap = Units(
                    value=Units(value=ref_net_cap, unit=ref_cap_unit).conversion(
                        new_unit=indoor_fan_power_unit
                    )
                    + Units(
                        value=indoor_fan_power, unit=indoor_fan_power_unit
                    ).conversion(new_unit="kW"),
                    unit="kW",
                ).conversion(ref_cap_unit)
        else:
            if ref_net_cap != None:
                logging.error("Input must be one and only one capacity input")
                raise ValueError("Input must be one and only one capacity input")
            if indoor_fan_power == None:
                # This is 400 cfm/ton and 0.365 W/cfm. Equation 11.1 from AHRI 210/240 (2024).
                indoor_fan_power_unit = "kW"
                indoor_fan_power = Units(
                    value=(
                        400
                        * 0.365
                        * Units(value=ref_gross_cap, unit=ref_cap_unit).conversion(
                            new_unit="ton"
                        )
                    )
                    / (
                        1
                        + 400
                        * 0.365
                        * Units(value=1.0, unit=ref_cap_unit).conversion(new_unit="ton")
                        * Units(value=1.0, unit="W").conversion(new_unit=ref_cap_unit)
                    ),
                    unit="W",
                ).conversion(new_unit=indoor_fan_power_unit)
                if not log_fan:
                    logging.info(f"Default fan power used: {indoor_fan_power} kW")
                    log_fan = True
            ref_net_cap = Units(
                value=Units(value=ref_gross_cap, unit=ref_cap_unit).conversion(
                    new_unit=indoor_fan_power_unit
                )
                - indoor_fan_power,
                unit=indoor_fan_power_unit,
            ).conversion(ref_cap_unit)
        self.ref_cap_unit = ref_cap_unit
        if self.ref_cap_unit != "kW":
            ref_net_cap_ton = Units(value=ref_net_cap, unit=self.ref_cap_unit)
            self.ref_net_cap = ref_net_cap_ton.conversion(new_unit="kW")
            ref_gross_cap_ton = Units(value=ref_gross_cap, unit=self.ref_cap_unit)
            self.ref_gross_cap = ref_gross_cap_ton.conversion(new_unit="kW")
            self.ref_cap_unit = "kW"
        else:
            self.ref_net_cap = ref_net_cap
            self.ref_gross_cap = ref_gross_cap
            self.ref_cap_unit = ref_cap_unit

        # Get attributes
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.full_eff_alt = 0
        self.full_eff_alt_unit = full_eff_unit
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.part_eff_alt = 0
        self.part_eff_alt_unit = part_eff_unit
        self.compressor_type = compressor_type
        self.set_of_curves = set_of_curves
        self.part_eff_ref_std = part_eff_ref_std
        self.model = model
        self.sim_engine = sim_engine
        self.part_eff_ref_std_alt = part_eff_ref_std_alt
        self.condenser_type = condenser_type
        self.compressor_speed = compressor_speed
        self.indoor_fan_speeds_mapping = indoor_fan_speeds_mapping
        self.indoor_fan_speeds = indoor_fan_speeds
        self.indoor_fan_power = indoor_fan_power
        self.indoor_fan_curve_coef = indoor_fan_curve_coef
        self.indoor_fan_power_unit = indoor_fan_power_unit
        self.indoor_fan_curve = indoor_fan_curve

        compressor_stages = sorted(compressor_stages)
        if len(compressor_stages) == 0:
            compressor_stages = [1]
        else:
            if compressor_stages[-1] < 1:
                compressor_stages.append(1)
        self.compressor_stages = compressor_stages
        self.stages = str(len(self.compressor_stages))

        # Convert control power to kW
        if len(control_power) > 0:
            for stage, power in control_power.items():
                control_power[stage] = Units(
                    value=power, unit=control_power_unit
                ).conversion(new_unit="kW")
        self.control_power = control_power

        # Define rated temperatures
        # air entering drybulb, air entering wetbulb, entering condenser temperature, leaving condenser temperature
        _, self.aew, ect, _ = self.get_rated_temperatures()
        self.ect = ect[0]

        self.default_fan_curve = Curve(
            eqp=self, c_type=self.indoor_fan_curve_coef["type"]
        )
        self.default_fan_curve.coeff1 = self.indoor_fan_curve_coef["1"]
        self.default_fan_curve.coeff2 = self.indoor_fan_curve_coef["2"]
        self.default_fan_curve.coeff3 = self.indoor_fan_curve_coef["3"]
        self.default_fan_curve.coeff4 = self.indoor_fan_curve_coef["4"]

        # Defined plotting ranges and (rated) temperature for normalization
        nb_val = 50
        if self.model == "simplified_bf":
            self.plotting_range = {
                "eir-f-t": {
                    "x1_min": self.aew,
                    "x1_max": self.aew,
                    "x1_norm": self.aew,
                    "nbval": nb_val,
                    "x2_min": 15,
                    "x2_max": 40,
                    "x2_norm": self.ect,
                },
                "cap-f-t": {
                    "x1_min": self.aew,
                    "x1_max": self.aew,
                    "x1_norm": self.aew,
                    "nbval": 50,
                    "x2_min": 15,
                    "x2_max": 40,
                    "x2_norm": self.ect,
                    "nbval": nb_val,
                },
                "eir-f-ff": {"x1_min": 0, "x1_max": 2, "x1_norm": 1, "nbval": nb_val},
                "cap-f-ff": {"x1_min": 0, "x1_max": 2, "x1_norm": 1, "nbval": nb_val},
                "plf-f-plr": {"x1_min": 0, "x1_max": 1, "x1_norm": 1, "nbval": nb_val},
            }

        # Cycling degradation
        self.degradation_coefficient = degradation_coefficient
        self.add_cycling_degradation_curve()

    def add_cycling_degradation_curve(self, overwrite=False, return_curve=False):
        """Determine and assign a part load fraction as a function of part load ratio curve to a unitary DX equipment.

        :param str overwrite: Flag to overwrite the existing degradation curve. Default is False
        :param bool overwrite: Overwrite existing plf-f-plr curves
        :param bool return_curve: Return the curves
        """
        # Remove existing curve if it exists
        if overwrite:
            for curve in self.set_of_curves:
                if curve.out_var == "plf-f-plr":
                    self.set_of_curves.remove(curve)
                    break

        # Add new curve
        for stage, stage_curves in self.get_dx_curves().items():
            if "plf-f-plr" not in stage_curves.keys() or overwrite:
                plf_f_plr = Curve(eqp=self, c_type="linear")
                plf_f_plr.speed = stage
                plf_f_plr.out_var = "plf-f-plr"
                plf_f_plr.type = "linear"
                plf_f_plr.coeff1 = 1 - self.degradation_coefficient
                plf_f_plr.coeff2 = self.degradation_coefficient
                plf_f_plr.x_min = 0
                plf_f_plr.x_max = 1
                plf_f_plr.out_min = 0
                plf_f_plr.out_max = 1
                if return_curve:
                    return plf_f_plr
                else:
                    self.set_of_curves.append(plf_f_plr)

    def calc_fan_power(
        self, compressor_stage, provide_flow_fraction=False, flow_fraction=1
    ):
        """Calculate unitary DX equipment fan power.

        :param float compressor_stage: Compressor stage associated with a specific fan speed
        :param bool provide_flow_fraction: Flag that indicates if the flow fraction should be returned
        :param float flow_fraction: Default flow fraction
        :return: Unitary DX Equipment fan power in Watts, (and flow fraction)
        :rtype: float

        """
        # Full flow/power
        compressor_stage = float(compressor_stage)
        if (
            compressor_stage == len(self.compressor_stages)
            or self.indoor_fan_speeds == 1
        ):
            flow_fraction = 1.0
            if provide_flow_fraction:
                return self.indoor_fan_power, flow_fraction
            else:
                return self.indoor_fan_power
        else:
            if self.indoor_fan_curve == False:
                compressor_stages = []
                fan_power_fractions = []
                fan_flow_fractions = []
                for speed_info in self.indoor_fan_speeds_mapping.values():
                    compressor_stages.append(speed_info["compressor_stage"])
                    fan_power_fractions.append(speed_info["fan_power_fraction"])
                    fan_flow_fractions.append(speed_info["fan_flow_fraction"])
                if compressor_stage <= compressor_stages[0]:  # Minimum flow/power
                    if provide_flow_fraction:
                        return (
                            self.indoor_fan_power * fan_power_fractions[0],
                            fan_flow_fractions[0],
                        )
                    else:
                        return self.indoor_fan_power * fan_power_fractions[0]

                elif compressor_stage in compressor_stages:
                    if provide_flow_fraction:
                        return (
                            self.indoor_fan_power
                            * fan_power_fractions[
                                compressor_stages.index(compressor_stage)
                            ],
                            fan_power_fractions[
                                compressor_stages.index(compressor_stage)
                            ],
                        )
                    else:
                        return (
                            self.indoor_fan_power
                            * fan_power_fractions[
                                compressor_stages.index(compressor_stage)
                            ]
                        )
                else:
                    # In between-stages
                    for i, ratio in enumerate(compressor_stages):
                        if (
                            ratio < compressor_stage
                            and compressor_stages[i + 1] > compressor_stage
                        ):
                            a = (
                                fan_power_fractions[i + 1] - fan_power_fractions[i]
                            ) / (compressor_stages[i + 1] - compressor_stages[i])
                            b = fan_power_fractions[i] - a * compressor_stages[i]
                            c = (fan_flow_fractions[i + 1] - fan_flow_fractions[i]) / (
                                compressor_stages[i + 1] - compressor_stages[i]
                            )
                            d = fan_flow_fractions[i] - c * compressor_stages[i]
                            if provide_flow_fraction:
                                return (
                                    self.indoor_fan_power * (a * compressor_stage + b),
                                    c * compressor_stage + d,
                                )
                            else:
                                return self.indoor_fan_power * (
                                    a * compressor_stage + b
                                )
            else:  # using curve
                default_min_fan_power = (
                    self.indoor_fan_power * 0.25
                )  # default min fan power
                power_factor = self.default_fan_curve.evaluate(x=flow_fraction, y=0)
                if self.indoor_fan_power * power_factor > default_min_fan_power:
                    if provide_flow_fraction:
                        return self.indoor_fan_power * power_factor, 1.0
                    else:
                        return self.indoor_fan_power * power_factor
                else:
                    if provide_flow_fraction:
                        return default_min_fan_power, 1.0
                    else:
                        return default_min_fan_power

    def calc_rated_eff(
        self,
        eff_type="part",
        unit="cop",
        output_report=False,
        alt=False,
        apply_modifiers_at_full_load=False,
    ):
        """Calculate unitary DX equipment efficiency.

        :param str eff_type: Unitary DX equipment efficiency type, currently supported `full` (full load rating)
                            and `part` (part load rating)
        :param str unit: Efficiency unit
        :param bool output_report: Indicate output report generation
        :param bool alt: Indicate the DX system alternate standard rating should be used
        :param bool apply_modifiers_at_full_load: Indicate if capacity and EIR modifiers at rated conditions will be applied at 100% load, False assumes that the performance curves are normalized at the rating conditions
        :return: Unitary DX Equipment rated efficiency
        :rtype: float

        """

        # Handle alternate ratings (not currently used)
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std

        # Get curves
        high_stage_id = self.stages
        high_stage_id_str = str(high_stage_id)
        curves = self.get_dx_curves()

        # Check if PLR curves exists, if not, add default curves
        plf_curves_found = False
        for stage_curves in curves.values():
            if "plf-f-plr" in stage_curves.keys():
                plf_curves_found = True
                break
        if not plf_curves_found:
            self.add_cycling_degradation_curve()
        curves = self.get_dx_curves()

        # Calculate capacity and efficiency degradation as a function of flow fraction
        tot_cap_flow_mod_fac = curves[high_stage_id_str]["cap-f-ff"].evaluate(1, 1)
        eir_flow_mod_fac = curves[high_stage_id_str]["eir-f-ff"].evaluate(1, 1)

        # Get rated conditions
        eqp_type = self.type.lower()
        equipment_type = equipment_references[eqp_type][std]
        load_fractions = equipment_type["coef"]["load_fractions"]
        weighting_factor = equipment_type["coef"]["weightingfactor"]
        eawbt = Equipment.convert_to_deg_c(equipment_type[self.condenser_type]["aew"])
        oabdt = Equipment.convert_to_deg_c(
            equipment_type[self.condenser_type]["ect"][0]
        )
        tot_cap_temp_mod_fac = curves[high_stage_id_str]["cap-f-t"].evaluate(
            eawbt, oabdt
        )
        eir_temp_mod_fac = curves[high_stage_id_str]["eir-f-t"].evaluate(eawbt, oabdt)
        if apply_modifiers_at_full_load:
            cap_modifiers = tot_cap_temp_mod_fac * tot_cap_flow_mod_fac
            eir_modifiers = eir_temp_mod_fac * eir_flow_mod_fac
        else:
            cap_modifiers = 1
            eir_modifiers = 1

        # Calculate NET rated capacity
        net_cooling_cap_rated = (
            self.ref_gross_cap * cap_modifiers - self.indoor_fan_power
        )

        # Convert user-specified full load efficiency to COP
        # User-specified efficiency is a NET efficiency
        full_eff = Units(value=self.full_eff, unit=self.full_eff_unit)
        rated_cop = full_eff.conversion(new_unit="cop")
        control_power = 0
        if len(self.control_power) > 0:
            control_power = self.control_power[high_stage_id_str]
        if rated_cop > 0.0:
            net_power = net_cooling_cap_rated / rated_cop
            gross_power = (
                net_power - self.indoor_fan_power - control_power
            )  # gross_power = compressor power + condenser power (outdoor fan)
            gross_cop = self.ref_gross_cap / gross_power
            gross_eir = eir_modifiers / gross_cop
            if output_report:
                logging.info(
                    f"Net cooling capacity at rated conditions: {net_cooling_cap_rated} W"
                )
                logging.info(f"Net rated COP at rated conditions: {rated_cop}")
                logging.info(f"Gross power at rated conditions: {gross_power} W")
        else:
            gross_eir = 0.0
            logging.error("Input COP is 0")
            raise ValueError("Input COP is 0")

        # Iterate through the different sets of rating conditions to calculate IEER
        ieer = 0
        for id, load_fraction in enumerate(load_fractions):
            # Load on the system
            load = load_fraction * net_cooling_cap_rated

            # Get the rating temperatures
            _, eawbt, _, _ = self.get_rated_temperatures()
            oabdt = self.get_rating_outdoor_unit_inlet_air_dry_bulb_temperature(
                load_fraction
            )

            # Determine intermediate efficiency calculation approach
            # - full_load: the unit has to run without having to cycle in order to meet the load
            # - degradation: the unit cycles to meet the load, the calculation is done using AHRI's coefficient of degradation, see Section 6.2.3.2 in AHRI 340/360 (2022)
            # - interpolation: the unit will cycle in between speeds to meet the load, the calculation is done using AHRI's, see Section 6.2.3.1 in AHRI 340/360 (2022)
            if load_fraction < 1:
                if len(self.compressor_stages) == 0:
                    capacity_ratio = 1
                else:
                    capacity_ratio = self.compressor_stages[0]

                lowest_stage_capacity = (
                    capacity_ratio
                    * net_cooling_cap_rated
                    * curves["1"]["cap-f-ff"].evaluate(1, 1)
                    * curves["1"]["cap-f-t"].evaluate(eawbt, oabdt)
                )

                if load <= lowest_stage_capacity:
                    calculation_approach = "degradation"
                else:
                    calculation_approach = "interpolation"
            else:
                calculation_approach = "full_load"

            # Calculate intermediate EER
            if calculation_approach == "degradation":
                intermediate_eer, _, _ = self.calculate_intermediate_eer(
                    curves,
                    load_fraction,
                    "1",
                    gross_eir,
                    report=output_report,
                    apply_modifiers_at_full_load=apply_modifiers_at_full_load,
                )
            elif calculation_approach == "full_load":
                intermediate_eer, _, _ = self.calculate_intermediate_eer(
                    curves,
                    load_fraction,
                    str(len(self.compressor_stages)),
                    gross_eir,
                    report=output_report,
                    apply_modifiers_at_full_load=apply_modifiers_at_full_load,
                )
            elif calculation_approach == "interpolation":
                intermediate_eer = self.calculate_intermediate_eer_by_interpolation(
                    net_cooling_cap_rated,
                    curves,
                    load,
                    load_fraction,
                    gross_eir,
                    report=output_report,
                )
            else:
                logging.error(
                    f"The intermediate efficiency calculation approach could not be determined for a {load_fraction} load fraction."
                )
                raise ValueError

            # Stop the process if only the full load efficiency is needed
            if eff_type == "full":
                ieer = intermediate_eer
                break

            if output_report:
                eer = Units(value=intermediate_eer, unit="cop")
                eer = eer.conversion(new_unit=self.full_eff_unit)
                logging.info(
                    f"EER at {load_fraction} AHRI load fraction: {eer.round(2)}\n"
                )

            # Update IEER
            ieer += weighting_factor[id] * intermediate_eer

        # Convert efficiency to original unit unless specified
        if unit != "cop":
            ieer = Units(value=ieer, unit="cop")
            ieer = ieer.conversion(new_unit=self.full_eff_unit)

        return ieer

    def get_rating_outdoor_unit_inlet_air_dry_bulb_temperature(self, percent_load):
        """Determine for a specific percent load percent the outdoor unit inlet air dry-bulb temperature.

        :param float percent_load: Efficiency unit
        :return: Outdoor unit inlet air dry-bulb temperature
        :rtype: float

        """
        if percent_load > 0.444:
            temperature = 5.0 + 30.0 * percent_load
        else:
            temperature = Equipment.convert_to_deg_c(
                equipment_references["unitarydirectexpansion"][self.part_eff_ref_std][
                    self.condenser_type
                ]["ect"][3]
            )
        return temperature

    def calculate_intermediate_eer_by_interpolation(
        self,
        net_cooling_cap_rated,
        curves,
        load,
        load_fraction,
        gross_eir,
        report=False,
        apply_modifiers_at_full_load=False,
    ):
        """Calculate intermediate EER (as part of the IEER calculation) by interpolation as per AHRI, see Section 6.2.3.1 in AHRI 340/360 (2022).

        :param float net_cooling_cap_rated: Net rated cooling capacity
        :param dict curves: Performance curves associated with the unit
        :param float load: Load corresponding to the load fraction for the IEER calculation
        :param float load_fraction: Load fraction used for the IEER calculation: 1, 0.75, 0.5, or 0.25
        :param float gross_eir: The unit's gross (only compressor and condenser section power) energy recovery ratio (1/COP)
        :param bool report: Flag that indicates if intermediate calculated variables should be logged
        :param bool apply_modifiers_at_full_load: Flag that defines if the capacity and EIR modifiers should be applied when the unit's performance is calculated for a load_fraction = 1
        :return: Energy Efficiency Ration (EER)
        :rtype: float

        """
        for stage_id, capacity_ratio in enumerate(self.compressor_stages):
            # Get the rating temperatures
            _, eawbt, _, _ = self.get_rated_temperatures()
            oabdt = self.get_rating_outdoor_unit_inlet_air_dry_bulb_temperature(
                load_fraction
            )

            # Capacity at current stage
            current_stage = stage_id + 1  # stages uses a 1-based indexing
            current_stage_str = str(current_stage)
            stage_capacity = (
                capacity_ratio
                * net_cooling_cap_rated
                * curves[current_stage_str]["cap-f-ff"].evaluate(1, 1)
                * curves[current_stage_str]["cap-f-t"].evaluate(eawbt, oabdt)
            )

            # Capacity at next stage, if available
            next_stage = stage_id + 2
            next_stage_str = str(next_stage)
            if next_stage <= len(self.compressor_stages):
                next_capacity_ratio = self.compressor_stages[
                    stage_id + 1
                ]  # compressor stages uses a 0-based indexing
                next_stage_capacity = (
                    next_capacity_ratio
                    * net_cooling_cap_rated
                    * curves[next_stage_str]["cap-f-ff"].evaluate(1, 1)
                    * curves[next_stage_str]["cap-f-t"].evaluate(eawbt, oabdt)
                )

                # Check if interpolation is required
                if stage_capacity < load < next_stage_capacity:
                    stage_id_low = current_stage
                    stage_id_high = next_stage
                    break
                else:
                    if next_stage == len(self.compressor_stages):
                        logging.error(
                            f"The load {load} is either greater than then capacity at stage {current_stage} ({stage_capacity} kW) or larger than the capacity at stage {next_stage} ({next_stage_capacity} kW)."
                        )
                        raise ValueError("Interpolation cannot be performed")
                    else:
                        next
        # Perform interpolation
        eer_low, _, actual_load_low = self.calculate_intermediate_eer(
            curves,
            load / net_cooling_cap_rated,
            stage_id_low,
            gross_eir,
            report,
            apply_modifiers_at_full_load,
            degradation=False,
        )
        eer_high, _, actual_load_high = self.calculate_intermediate_eer(
            curves,
            load / net_cooling_cap_rated,
            stage_id_high,
            gross_eir,
            report,
            apply_modifiers_at_full_load,
            degradation=False,
        )
        intermediate_eer = (
            (eer_high - eer_low) / (actual_load_high - actual_load_low)
        ) * (load_fraction - actual_load_low) + eer_low
        if report:
            logging.info(
                f"EER at current stage {current_stage}: {Units(value=eer_low, unit='cop').conversion(new_unit='eer')}"
            )
            logging.info(
                f"EER at next stage stage: {Units(value=eer_high, unit='cop').conversion(new_unit='eer')}"
            )
        return intermediate_eer

    def calculate_intermediate_eer(
        self,
        curves,
        load_fraction,
        stage,
        gross_eir,
        report,
        apply_modifiers_at_full_load,
        degradation=True,
    ):
        """Calculate intermediate EER (as part of the IEER calculation) using the degradation approach as per Section 6.2.3.2 in AHRI 340/360 (2022).

        :param dict curves: Performance curves associated with the unit
        :param float load_fraction: Load fraction used for the IEER calculation: 1, 0.75, 0.5, or 0.25
        :param int stage: Current stage
        :param float gross_eir: The unit's gross (only compressor and condenser section power) energy recovery ratio (1/COP)
        :param bool report: Flag that indicates if intermediate calculated variables should be logged
        :param bool apply_modifiers_at_full_load: Flag that defines if the capacity and EIR modifiers should be applied when the unit's performance is calculated for a load_fraction = 1
        :param bool degradation: Flag that indicates whether or not the degradation coefficient should be applied
        :return: Energy Efficiency Ration (EER), Load factor (LF), Actual load
        :rtype: list

        """
        # Calculate capacity at rating conditions
        _, eawbt, _, _ = self.get_rated_temperatures()
        oabdt = self.get_rating_outdoor_unit_inlet_air_dry_bulb_temperature(
            load_fraction
        )

        # Determine the fan power
        # First pass: assumes no performance impact of airflow
        current_stage = str(stage)
        tot_cap_temp_mod_fac = curves[current_stage]["cap-f-t"].evaluate(eawbt, oabdt)
        tot_cap_flow_mod_fac = curves[current_stage]["cap-f-ff"].evaluate(1.0, 1.0)
        indoor_fan_power, flow_fraction = self.calc_fan_power(
            current_stage, True, report
        )

        # Second pass: assume performance impact of airflow now that the airflow fraction is known
        tot_cap_flow_mod_fac = curves[current_stage]["cap-f-ff"].evaluate(
            flow_fraction, 1.0
        )
        indoor_fan_power, flow_fraction = self.calc_fan_power(current_stage, True)

        # Determine the net cooling capacity for this stage at the rating condition corresponding to the IEER load fraction
        if not apply_modifiers_at_full_load and load_fraction == 1:
            net_cooling_cap = (
                self.ref_gross_cap * self.compressor_stages[int(current_stage) - 1]
                - indoor_fan_power
            )
        else:
            net_cooling_cap = (
                self.ref_gross_cap
                * tot_cap_temp_mod_fac
                * tot_cap_flow_mod_fac
                * self.compressor_stages[int(current_stage) - 1]
                - indoor_fan_power
            )

        # Calculate the load factor (not the same as the load fraction) as per AHRI 340/360
        if degradation:
            load_factor = min(1.0, load_fraction * self.ref_net_cap / net_cooling_cap)
        else:
            load_factor = 1.0

        # Calculate efficiency at rating conditions
        eir_temp_mod_fac = curves[current_stage]["eir-f-t"].evaluate(eawbt, oabdt)
        eir_flow_mod_fac = curves[current_stage]["eir-f-ff"].evaluate(
            flow_fraction, 1.0
        )
        if not apply_modifiers_at_full_load and load_fraction == 1:
            eir = gross_eir
        else:
            eir = gross_eir * eir_temp_mod_fac * eir_flow_mod_fac

        if report:
            logging.info(f"Current stage: {stage}")
            logging.info(f"Flow fraction: {flow_fraction}")
            logging.info(
                f"Rating entering air wet-bulb temperature (in deg. C): {eawbt}"
            )
            logging.info(
                f"Rating outdoor air dry-bulb temperature (in deg. C): {oabdt}"
            )
            logging.info(f"Indoor fan power: {indoor_fan_power} kW")
            logging.info(
                f"Capacity modifier as a function of flow fraction: {tot_cap_flow_mod_fac}"
            )
            logging.info(
                f"Capacity modifier as a function of temperature: {tot_cap_temp_mod_fac}"
            )
            logging.info(f"Load factor (LF in AHRI 340/360): {load_factor}")
            logging.info(
                f"Compressor capacity ratio (net rated capacity at current stage to net rated cooling capacity at highest stage): {self.compressor_stages[int(stage) - 1]}"
            )
            logging.info(f"Net cooling capacity at this stage: {net_cooling_cap}")
            logging.info(
                f"EIR modifier as a function of flow fraction: {eir_flow_mod_fac}"
            )
            logging.info(
                f"EIR modifier as a function of temperature: {eir_temp_mod_fac}"
            )
            logging.info(f"The unit's gross reference EIR: {gross_eir}")
            logging.info(f"Gross EIR at this stage: {eir}")

        # Determine cycling degradation coefficient
        if degradation:
            if not apply_modifiers_at_full_load and load_fraction == 1:
                degradation_coeff = 1
            else:
                degradation_coeff = 1 / curves[current_stage]["plf-f-plr"].evaluate(
                    load_factor, 1
                )
            if report:
                logging.info(f"Degradation coefficient: {degradation_coeff}")
        else:
            degradation_coeff = 1.0

        # Determine Compressor power and outdoor fan (P_C + P_CD)
        if not apply_modifiers_at_full_load and load_fraction == 1:
            elec_power = (
                degradation_coeff
                * eir
                * (
                    self.ref_gross_cap
                    * 1
                    * self.compressor_stages[int(current_stage) - 1]
                )
            )
        else:
            elec_power = (
                degradation_coeff
                * eir
                * (
                    self.ref_gross_cap
                    * tot_cap_temp_mod_fac
                    * tot_cap_flow_mod_fac
                    * self.compressor_stages[int(current_stage) - 1]
                )
            )

        # Determine control power (P_CT)
        control_power = 0
        if len(self.control_power) > 0:
            if current_stage in self.control_power.keys():
                control_power = self.control_power[current_stage]

        # Determine load on the unit
        actual_load = net_cooling_cap / self.ref_net_cap

        # Calculate EER
        eer = (load_factor * net_cooling_cap) / (
            load_factor * elec_power + indoor_fan_power + control_power
        )

        if report:
            logging.info(f"Control power: {control_power} kW")
            logging.info(f"Compressor and condenser power: {elec_power} kW")
            logging.info(f"Actual load: {actual_load} kW")
            logging.info(
                f"EER at current stage {current_stage}: {Units(value=eer, unit='cop').conversion(new_unit='eer')}"
            )

        return eer, load_factor, actual_load

    def ieer_to_eer(self, ieer):
        """Calculate EER from IEER and system capacity.
        The regression function was obtained by fitting a linear model on performance data collected from AHRI database (Sample Size = 14,268).
        Model Diagnostics:
        R-square = 0.5458
        Mean Absolute Error = 0.369
        Root Mean Square Error = 0.455
        Model was internally validated using 10-fold cross validation approach and externally validated using the USDOE database.

        :param float ieer: Integrated energy efficiency ratio (IEER)
        :return: Energy efficiency ratio (EER)
        :rtype: float

        """

        ref_net_cap = Units(value=self.ref_net_cap, unit=self.ref_cap_unit).conversion(
            new_unit="btu/h"
        )

        eer = (
            9.886
            + 0.1804 * ieer
            - (1.88e-17) * (ref_net_cap**3)
            + (2.706e-11) * (ref_net_cap**2)
            - (1.047e-5) * (ref_net_cap)
            - (1.638e-7) * (ieer * ref_net_cap)
        )
        return eer

    def get_dx_curves(self, copy_all_stages=True):
        """Retrieve DX curves from the DX set_of_curves attribute.

        :return: Dictionary of the curves associated with the object
        :rtype: dict

        """
        curves = {}
        for s in range(int(self.stages)):
            curves[f"{s +1}"] = {}
        for curve in self.set_of_curves:
            if int(curve.speed) <= int(self.stages):
                if curve.out_var == "cap-f-t":
                    curves[str(curve.speed)]["cap-f-t"] = curve
                if curve.out_var == "cap-f-ff":
                    curves[str(curve.speed)]["cap-f-ff"] = curve
                if curve.out_var == "eir-f-t":
                    curves[str(curve.speed)]["eir-f-t"] = curve
                if curve.out_var == "eir-f-ff":
                    curves[str(curve.speed)]["eir-f-ff"] = curve
                if curve.out_var == "plf-f-plr":
                    curves[str(curve.speed)]["plf-f-plr"] = curve

        if (
            int(self.stages) > 1 and copy_all_stages
        ):  # Add curves to other stages after aggregation
            for ct in ["cap-f-t", "cap-f-ff", "eir-f-t", "eir-f-ff", "plf-f-plr"]:
                for s in curves.keys():
                    if ct not in curves[s].keys() and ct in curves["1"]:
                        curves[s][ct] = curves["1"][ct]

        return curves

    def get_curves_from_lib(self, lib, filters):
        """Function to get the sort from the library based on chiller filters.

        :param copper.library.Library lib: Chiller library object
        :param list filters: List of tuples containing the relevant filter keys and values
        :return: List of set of curves object corresponding to seed curves
        :rtype: list

        """
        sets = lib.find_set_of_curves_from_lib(filters=filters, part_eff_flag=True)
        assert sets is not None
        return sets

    def get_rated_temperatures(self, alt=False):
        """Get unitary DX equipment rated temperatures.

        :param bool alt: Indicate the unitary DX equipment alternate standard rating should be used
        :return: Rated entering condenser temperature and temperature of the air entering the system
        :rtype: list

        """
        if alt:
            std = self.part_eff_ref_std_alt
        else:
            std = self.part_eff_ref_std
        dx_data = equipment_references[self.type.lower()][std][self.condenser_type]
        # Air entering indoor dry-bulb
        aed = Equipment.convert_to_deg_c(
            dx_data["aed"], dx_data["ae_unit"]
        )  # TODO should be based on load
        # Air entering indoor wet-bulb
        self.aew = Equipment.convert_to_deg_c(dx_data["aew"], dx_data["ae_unit"])
        # Outdoor water/air entering
        ect = [
            Equipment.convert_to_deg_c(t, dx_data["ect_unit"]) for t in dx_data["ect"]
        ]
        # Outdoor water/air leaving
        lct = Equipment.convert_to_deg_c(dx_data["lct"], dx_data["lct_unit"])
        return [aed, self.aew, ect, lct]

    def get_lib_and_filters(self, lib_path=unitary_dx_lib):
        """Get unitary DX equipment library object and unitary DX equipment specific filters.

        :param str lib_path:Full path of json library
        :return: Unitary DX equipment library object and filters
        :rtype: list

        """
        lib = Library(path=lib_path)
        filters = [
            ("eqp_type", "UnitaryDirectExpansion"),
            ("condenser_type", self.condenser_type),
            ("sim_engine", self.sim_engine),
            ("model", self.model),
        ]

        return lib, filters

    def get_ranges(self):
        """Get applicable range of values for independent variables of the unitary DX equipment model.

        :return: Range of values, and values used for normalization (reference/rated values)
        :rtype: dict

        """
        ranges = {
            "eir-f-t": {
                "vars_range": [(12.8, 26), (10.0, 40.0)],
                "normalization": (self.aew, self.ect),
            },
            "eir-f-ff": {"vars_range": [(0.0, 1.5)], "normalization": (1.0)},
            "cap-f-t": {
                "vars_range": [(12.8, 26), (10.0, 40.0)],
                "normalization": (self.aew, self.ect),
            },
            "cap-f-ff": {"vars_range": [(0.0, 1.5)], "normalization": (1.0)},
            "plf-f-plr": {"vars_range": [(0.0, 1.0)], "normalization": (1.0)},
        }

        return ranges

    def get_seed_curves(self, lib=None, filters=None, csets=None):
        """Function to generate seed curves specific to a unitary DX equipment and sets relevant attributes (misc_attr, ranges).

        :param copper.library.Library lib: Unitary DX equipment library object
        :param list filters: List of tuples containing the filter keys and values
        :param list csets: List of set of curves object corresponding to selected unitary DX equipment from library
        :rtype: SetsofCurves

        """
        assert self.compressor_type in ["scroll"]

        if lib is None or filters is None or csets is None:
            lib, filters = self.get_lib_and_filters()
            csets = self.get_curves_from_lib(lib=lib, filters=filters)

        full_eff = Units(self.full_eff, self.full_eff_unit)
        full_eff_cop = full_eff.conversion("cop")
        part_eff = Units(self.part_eff, self.part_eff_unit)
        part_eff_cop = part_eff.conversion("cop")

        self.misc_attr = {
            "model": self.model,
            "ref_net_cap": self.ref_net_cap,
            "ref_gross_cap": self.ref_gross_cap,
            "full_eff": full_eff_cop,
            "part_eff": part_eff_cop,
            "ref_eff_unit": self.full_eff_unit,
            "compressor_type": self.compressor_type,
            "condenser_type": self.condenser_type,
            "compressor_speed": self.compressor_speed,
            "sim_engine": self.sim_engine,
            "name": "Aggregated set of curves",
            "source": "Copper",
        }

        self.ranges = self.get_ranges()
        curves = SetsofCurves(sets=csets, eqp=self)
        return curves

    def get_ref_vars_for_aggregation(self):
        return ["ref_net_cap", "full_eff", "part_eff"]
