import json
import pandas as pd
import numpy as np
import itertools
import psychrolib

# Set psychrometric unit system to SI
psychrolib.SetUnitSystem(psychrolib.SI)

# === CONFIGURATION ===
csv_file = 'AC_Perf_901_2022_65_to_135_11.55EER_14.8IEER.csv'
json_template_file = 'DX-Constant-Efficiency.RS0004.a205.json'
output_json_file = 'DX_Updated_STD205_Output.json'

# Nominal values, gt760 for default
nominal_capacity = 232057  # W, 792 kBtu/h
nominal_eer = 9.2         # Example EER (IP)
nominal_eir = 1 / nominal_eer
nominal_SHR = 0.7

# === UTILITY FUNCTIONS ===
def compute_wetbulb(Tdb_K, RH_frac, pressure_kPa):
    """Compute wet-bulb temperature in Celsius."""
    Tdb_C = Tdb_K - 273.15
    pressure_Pa = pressure_kPa * 1000
    return psychrolib.GetTWetBulbFromRelHum(Tdb_C, RH_frac, pressure_Pa)

def evaluate_curve(row, x1, x2=0):
    ctype = row['CurveUse'].lower()
    if ctype == 'bi_quad':
        return row['C0'] + row['C1'] * x1 + row['C2'] * x1**2 + row['C3'] * x2 + row['C4'] * x2**2 + row['C5'] * x1 * x2
    elif ctype == 'cubic':
        return row['C0'] + row['C1'] * x1 + row['C2'] * x1**2 + row['C3'] * x1**3
    elif ctype == 'quadratic':
        return row['C0'] + row['C1'] * x1 + row['C2'] * x1**2
    elif ctype == 'linear':
        return row['C0'] + row['C1'] * x1
    else:
        raise ValueError(f"Unsupported curve type: {ctype}")

def calculate_performance(
    cap_f_t_row, cap_f_flow_row, eir_f_t_row, eir_f_flow_row, plf_f_plr_row,
    x1, x2, flow_ratio, plr, nominal_capacity, nominal_eir
):
    cap_f_t = evaluate_curve(cap_f_t_row, x1, x2)
    cap_f_flow = evaluate_curve(cap_f_flow_row, flow_ratio)
    eir_f_t = evaluate_curve(eir_f_t_row, x1, x2)
    eir_f_flow = evaluate_curve(eir_f_flow_row, flow_ratio)
    plf = evaluate_curve(plf_f_plr_row, plr)
    gross_capacity = nominal_capacity * cap_f_t * cap_f_flow
    eir = nominal_eir * eir_f_t * eir_f_flow
    power = gross_capacity * eir * (plr / plf if plf > 0 else 1)
    return gross_capacity, 0.0, power  # sensible capacity is placeholder

# --- NEW: colleague-style ADP finder (line to saturation), using PsychroLib ---
def _cp_moist_air_J_per_kgK(w):
    return 1006.0 + 1860.0 * w

def _dewpoint_from_w(w, P_Pa):
    """
    Compute dewpoint [°C] from humidity ratio w [kg/kg] and pressure P [Pa].
    """
    # Partial vapor pressure from humidity ratio
    Pw = psychrolib.GetVapPresFromHumRatio(w, P_Pa)
    # Need a dry-bulb guess as the first arg (PsychroLib ignores it, but must be >=0).
    # Safe to use 0.0 °C or any value.
    return psychrolib.GetTDewPointFromVapPres(0.0, Pw)

def estimate_sensible_capacity_coolpropline(
    Q_total_W, Tdbi_K, RH_frac, flow_rate_kg_s, pressure_kPa, SHR_rated=0.7
):
    """
    Colleague's approach:
      1) From Q_total and SHR, derive (T_out, w_out).
      2) Fit line (T, w) through (T_in, w_in) and (T_out, w_out).
      3) March along that line to find ADP where T_dp(w_x) ~= T_x.
      4) Compute sensible/latent using T_out from step 1.

    Returns: Q_sensible_W, Q_latent_W, SHR_actual
    """
    P_Pa = pressure_kPa * 1000.0
    T_in_C = Tdbi_K - 273.15
    RH = max(0.01, min(0.99, RH_frac))
    m_dot = max(flow_rate_kg_s, 1e-9)

    # Inlet state
    w_in = psychrolib.GetHumRatioFromRelHum(T_in_C, RH, P_Pa)
    h_in = psychrolib.GetMoistAirEnthalpy(T_in_C, w_in)

    # Outlet conditions from total capacity + SHR
    delta_h = Q_total_W / m_dot
    h_out = h_in - delta_h
    # Compute w_out using h at inlet T (standard SHR split)
    h_tin_wout = h_in - (1.0 - SHR_rated) * delta_h
    w_out = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h_tin_wout, T_in_C)
    # T_out from enthalpy & w_out
    T_out_C = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h_out, w_out)

    # Line in (T, w): w = a*T + b through (T_in, w_in) & (T_out, w_out)
    dT = T_out_C - T_in_C
    if abs(dT) < 1e-9:
        # near-zero temp drop ⇒ mostly sensible=0; return safe split
        cp_air = _cp_moist_air_J_per_kgK(w_in)
        Q_sens = m_dot * cp_air * (T_in_C - T_out_C)
        Q_lat = Q_total_W - Q_sens
        SHR_actual = Q_sens / max(Q_total_W, 1e-6)
        return Q_sens, Q_lat, SHR_actual

    a = (w_out - w_in) / dT
    b = w_in - a * T_in_C

    # Iteratively locate ADP along the line (match dewpoint to its temperature)
    # Start above T_out to avoid division issues
    t_x = T_out_C + 1e-3
    incr = 0.001
    for _ in range(2000):
        t_x += incr
        w_x = a * t_x + b
        # Guard bounds for humidity ratio
        w_x = max(1e-8, w_x)
        # Dewpoint (from w_x at pressure)
        t_dp = _dewpoint_from_w(w_x, P_Pa)
        err = t_dp - t_x
        if abs(err) < 1e-4:
            break
        # "CoolProp-style" step control
        incr = err / 10.0

    T_adp = t_x
    w_adp = a * T_adp + b
    w_adp = max(1e-8, w_adp)

    # Sensible / latent using T_out from step 1 (keeps total consistent)
    cp_air = _cp_moist_air_J_per_kgK(w_in)
    Q_sens = m_dot * cp_air * (T_in_C - T_out_C)
    Q_lat  = Q_total_W - Q_sens
    SHR_actual = Q_sens / max(Q_total_W, 1e-6)

    return Q_sens, Q_lat, SHR_actual

# === STEP 1: Load CSV ===
df = pd.read_csv(csv_file, header=None)
df.columns = [
    'CurveName', 'CurveType', 'Unused', 'CurveUse',
    'X1Min', 'X1Max', 'X2Min', 'X2Max',
    'C0', 'C1', 'C2', 'C3', 'C4', 'C5'
]

# === STEP 2: Load JSON Template ===
with open(json_template_file, 'r') as f:
    data = json.load(f)

grid = data['performance']['performance_map_cooling']['grid_variables']
keys = list(grid.keys())
values = [grid[k] for k in keys]
combinations = list(itertools.product(*values))

# === STEP 3: Identify Curve Rows ===
cap_f_t_row = df[df['CurveType'] == 'cap-f-t'].iloc[0]
cap_f_flow_row = df[df['CurveType'] == 'cap-f-ff'].iloc[0]
eir_f_t_row = df[df['CurveType'] == 'eir-f-t'].iloc[0]
eir_f_flow_row = df[df['CurveType'] == 'eir-f-ff'].iloc[0]
plf_f_plr_row = df[df['CurveType'] == 'plf-f-plr'].iloc[0]

# === STEP 4: Loop Through Grid ===
lookup = {
    "gross_total_capacity": [],
    "gross_sensible_capacity": [],
    "gross_power": [],
    "operation_state": []
}

for combo in combinations:
    combo_dict = dict(zip(keys, combo))
    try:
        Tdbi_K = combo_dict["indoor_coil_entering_dry_bulb_temperature"]
        RH_frac = max(0.01, min(1.0, combo_dict["indoor_coil_entering_relative_humidity"]))
        Tdbo_K = combo_dict["outdoor_coil_entering_dry_bulb_temperature"]
        P_kPa = combo_dict["ambient_absolute_air_pressure"]

        Tdbo_C = Tdbo_K - 273.15
        WBi_C = compute_wetbulb(Tdbi_K, RH_frac, P_kPa)

        flow_rate = combo_dict["indoor_coil_air_mass_flow_rate"]
        flow_ratio = flow_rate / max(grid["indoor_coil_air_mass_flow_rate"])
        plr = 1.0  # currently fixed

        gross_cap, _, power = calculate_performance(
            cap_f_t_row, cap_f_flow_row, eir_f_t_row, eir_f_flow_row, plf_f_plr_row,
            WBi_C, Tdbo_C, flow_ratio, plr,
            nominal_capacity, nominal_eir
        )

        # --- Apply compressor sequence degradation ---
        comp_stage = combo_dict.get("compressor_sequence_number", 1)
        if comp_stage > 1:
            # Assume half performance for stage >= 2
            degradation_factor = 0.5
            gross_cap *= degradation_factor
            power *= degradation_factor

        # --- Sensible/latent using colleague's ADP method (per point) ---
        sens_cap, _, _ = estimate_sensible_capacity_coolpropline(
            Q_total_W=gross_cap,
            Tdbi_K=Tdbi_K,
            RH_frac=RH_frac,
            flow_rate_kg_s=flow_rate,
            pressure_kPa=P_kPa,
            SHR_rated=nominal_SHR
        )

        lookup["gross_total_capacity"].append(gross_cap)
        lookup["gross_sensible_capacity"].append(sens_cap)
        lookup["gross_power"].append(power)
        lookup["operation_state"].append("NORMAL")

    except Exception as e:
        print(f"Skipping point {combo_dict} due to error: {e}")
        continue

# === STEP 5: Update JSON ===
data['performance']['performance_map_cooling']['lookup_variables'] = lookup

# === STEP 6: Save Updated JSON ===
with open(output_json_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ STD205 JSON updated and saved to: {output_json_file}")
