# ================================================
# Unitary DX Performance Mapping Generator for STD 205
# ================================================
# 1. Generate performance curves (CSV)
# 2. Populate STD205 JSON template with curves
# 3. Validate JSON against ASHRAE 205 schema
# 4. Convert JSON to other formats using tk205
# ================================================

import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psychrolib
from pathlib import Path
from jsonschema import Draft7Validator, RefResolver
import tk205
import os
import glob
import sys
# -------------------------
# STEP 1: Generate performance CSV
# -------------------------
"""
Generate performance curve CSVs for unitary DX equipment using the `copper` library.
Replicates the Jupyter notebook workflow (no plots, no IDF export).
"""
copper_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if copper_path not in sys.path:
    sys.path.insert(0, copper_path)
import copper as cp

random_seed = 1

capacities = {
   "65_to_135": 96,   # kBtu/h; 8 ton
   "135_to_240": 180, # kBtu/h; 15 ton
   "240_to_760": 480, # kBtu/h; 40 ton
   "gt760": 792,      # kBtu/h; 66 ton
}

# See backup calcs workbooks in kW
fan_power = {
   "65_to_135": 0.524,
   "135_to_240": 1.197,
   "240_to_760": 5.243,
   "gt760": 11.190,
}

requirements = {
    "901_2004": {
        "eer": {
            "65_to_135": 10.3,
            "135_to_240": 9.7,
            "240_to_760": 9.5,
            "gt760": 9.2,
        },
        "ieer": {
            "65_to_135": None,
            "135_to_240": None,
            "240_to_760": 9.7,
            "gt760": 9.4,
        }
    },
    "901_2022": {
        "eer": {
            "65_to_135": None,
            "135_to_240": None,
            "240_to_760": None,
            "gt760": None,
        },
        "ieer": {
            "65_to_135": 14.8,
            "135_to_240": 14.2,
            "240_to_760": 13.2,
            "gt760": 12.5,
        }
    }
}

for code in list(requirements.keys()):
    for cap in list(capacities.keys()):
        if cap not in requirements[code]["ieer"].keys():
            continue

        # Convert to ton
        tonnage = cp.Units(value=capacities[cap], unit="kbtu/h").conversion(new_unit="ton")

        # Indoor fan speeds per your logic
        if "2004" in code:
            if cap == "gt760":
                indoor_fan_speeds = 2
            else:
                indoor_fan_speeds = 1
        else:
            indoor_fan_speeds = 2

        # Build equipment object
        dx = cp.UnitaryDirectExpansion(
            compressor_type="scroll",
            condenser_type="air",
            compressor_speed="constant",
            ref_cap_unit="ton",
            ref_net_cap=tonnage,
            full_eff=requirements[code]["eer"][cap],
            full_eff_unit="eer",
            part_eff_ref_std="ahri_340/360",
            indoor_fan_speeds=indoor_fan_speeds,
            indoor_fan_power=fan_power[cap],
            indoor_fan_power_unit="kW",
        )

        # Branching consistent with your approach
        agg_only = False
        eer_val = requirements[code]["eer"][cap]
        ieer_val = requirements[code]["ieer"][cap]

        if eer_val is None:
            # Compute EER from IEER
            dx.part_eff = ieer_val
            dx.full_eff = dx.ieer_to_eer(ieer_val)
            name = f"AC_Perf_{code}_{cap}_{round(dx.full_eff,2)}EER_{round(dx.part_eff,2)}IEER"

        elif ieer_val is None:
            # EER-only case
            agg_only = True
            name = f"AC_Perf_{code}_{cap}_{round(dx.full_eff,2)}EER"

        else:
            # Both provided
            dx.part_eff = ieer_val
            name = f"AC_Perf_{code}_{cap}_{round(dx.full_eff,2)}EER_{round(dx.part_eff,2)}IEER"

        # Degradation & curve gen
        if "2004" in code:
            dx.degradation_coefficient = 0.25
        else:
            dx.degradation_coefficient = 0.15
        dx.add_cycling_degradation_curve(overwrite=True)

        _ = dx.generate_set_of_curves(
            method="nearest_neighbor",
            tol=0.01,
            num_nearest_neighbors=5,
            verbose=False,
            vars=["eir-f-t"],
            random_seed=random_seed,
            agg_only=agg_only
        )

        # Apply degradation again (matches original flow)
        dx.add_cycling_degradation_curve(overwrite=True)

        # Prepare curve set & enforce limits
        curves = cp.SetofCurves()
        curves.curves = dx.set_of_curves
        curves.eqp = dx

        limits = dx.get_ranges()
        for c in curves.curves:
            xs = limits[c.out_var]["vars_range"][0]
            c.x_min = xs[0]
            c.x_max = xs[1]
            if len(limits[c.out_var]["vars_range"]) > 1:
                ys = limits[c.out_var]["vars_range"][1]
                c.y_min = ys[0]
                c.y_max = ys[1]
            if "ff" in c.out_var:
                c.x_min = 0.4
            if "plf" in c.out_var:
                c.out_min = 0.0

        # Export each set to CSV (same naming)
        curves.export(path="./", fmt="csv", name=name)

# Combine into a single CSV (same header & approach as yours)
with open("./ieer_specific_curves.csv", "w") as nf:
    nf.write("name,variable,unit_type,curve_type,min_x,max_x,min_y,max_y,coeff1,coeff2,coeff3,coeff4,coeff5,coeff6\n")
    for f in glob.glob("./AC_Perf*.csv"):
        with open(f, "r") as curve_set:
            for line in curve_set:
                nf.write(line)


# -------------------------
# STEP 2: Populate STD205 JSON
# -------------------------
# Set psychrometric unit system to SI
psychrolib.SetUnitSystem(psychrolib.SI)
# === CONFIGURATION ===
csv_file = 'AC_Perf_901_2022_65_to_135_11.55EER_14.8IEER.csv' # example curve file
json_template_file = 'DX-Constant-Efficiency.RS0004.a205.json'
# Keep output under input/ so STEP 4 can convert it
output_json_file = 'input/DX_Updated_STD205_Output.json'

# Nominal values (example uses gt760 default capacity)
nominal_capacity = 232057  # W, ≈792 kBtu/h
nominal_eer = 9.2          # IP EER example
nominal_eir = 1.0 / max(nominal_eer, 1e-9)
nominal_SHR = 0.7

# === UTILITY FUNCTIONS ===
def compute_wetbulb(Tdb_K, RH_frac, pressure_kPa):
    """Compute wet-bulb temperature in Celsius."""
    Tdb_C = Tdb_K - 273.15
    pressure_Pa = pressure_kPa * 1000
    return psychrolib.GetTWetBulbFromRelHum(Tdb_C, RH_frac, pressure_Pa)

def evaluate_curve(row, x1, x2=0.0):
    ctype = str(row['CurveUse']).lower()
    if ctype == 'bi_quad':
        return (row['C0'] + row['C1'] * x1 + row['C2'] * x1**2
                + row['C3'] * x2 + row['C4'] * x2**2 + row['C5'] * x1 * x2)
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
    cap_f_t   = evaluate_curve(cap_f_t_row,   x1, x2)
    cap_f_ff  = evaluate_curve(cap_f_flow_row, flow_ratio)
    eir_f_t   = evaluate_curve(eir_f_t_row,   x1, x2)
    eir_f_ff  = evaluate_curve(eir_f_flow_row, flow_ratio)
    plf       = max(evaluate_curve(plf_f_plr_row, plr), 1e-6)

    gross_capacity = nominal_capacity * cap_f_t * cap_f_ff
    eir            = nominal_eir * eir_f_t * eir_f_ff
    power          = gross_capacity * eir * (plr / plf)
    return gross_capacity, 0.0, power  # sensible is computed via ADP method below

# --- colleague-style ADP finder (line to saturation), using PsychroLib ---
def _cp_moist_air_J_per_kgK(w):
    return 1006.0 + 1860.0 * w

def _dewpoint_from_w(w, P_Pa):
    """
    Dewpoint [°C] from humidity ratio w [kg/kg] and pressure P [Pa].
    Fixes the earlier error by providing both args to PsychroLib API.
    """
    Pw = psychrolib.GetVapPresFromHumRatio(w, P_Pa)
    # PsychroLib signature is GetTDewPointFromVapPres(TDryBulb, VapPres)
    # TDryBulb is not used in the calculation; a non-negative filler is fine.
    return psychrolib.GetTDewPointFromVapPres(0.0, Pw)

def estimate_sensible_capacity_coolpropline(
    Q_total_W, Tdbi_K, RH_frac, flow_rate_kg_s, pressure_kPa, SHR_rated=0.7
):
    """
    1) From Q_total and SHR, derive (T_out, w_out).
    2) Fit line (T, w) through (T_in, w_in) and (T_out, w_out).
    3) March along that line to find ADP where T_dp(w_x) ~= T_x.
    4) Compute sensible/latent using T_out (keeps Q_total consistent).
    """
    P_Pa = pressure_kPa * 1000.0
    T_in_C = Tdbi_K - 273.15
    RH = max(0.01, min(0.99, RH_frac))
    m_dot = max(flow_rate_kg_s, 1e-9)

    # Inlet state
    w_in = psychrolib.GetHumRatioFromRelHum(T_in_C, RH, P_Pa)
    h_in = psychrolib.GetMoistAirEnthalpy(T_in_C, w_in)

    # Outlet conditions from total capacity + SHR split
    delta_h = Q_total_W / m_dot
    h_out   = h_in - delta_h
    # Standard SHR split: keep T at inlet to solve w_out, then T_out from (h_out, w_out)
    h_tin_wout = h_in - (1.0 - SHR_rated) * delta_h
    w_out = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h_tin_wout, T_in_C)
    T_out_C = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h_out, w_out)

    # Line in (T, w)
    dT = T_out_C - T_in_C
    if abs(dT) < 1e-9:
        cp_air = _cp_moist_air_J_per_kgK(w_in)
        Q_sens = m_dot * cp_air * (T_in_C - T_out_C)
        Q_lat  = Q_total_W - Q_sens
        SHR_actual = Q_sens / max(Q_total_W, 1e-6)
        return Q_sens, Q_lat, SHR_actual

    a = (w_out - w_in) / dT
    b = w_in - a * T_in_C

    # March along line to find ADP (t_dp(w_x) ~ t_x)
    t_x = T_out_C + 1e-3
    incr = 0.001
    for _ in range(2000):
        t_x += incr
        w_x = max(1e-8, a * t_x + b)
        t_dp = _dewpoint_from_w(w_x, P_Pa)
        err = t_dp - t_x
        if abs(err) < 1e-4:
            break
        incr = err / 10.0

    # Sensible/latent with T_out from step above
    cp_air = _cp_moist_air_J_per_kgK(w_in)
    Q_sens = m_dot * cp_air * (T_in_C - T_out_C)
    Q_lat  = Q_total_W - Q_sens
    SHR_actual = Q_sens / max(Q_total_W, 1e-6)
    return Q_sens, Q_lat, SHR_actual

df_curves = pd.read_csv(csv_file, header=None)
df_curves.columns = [
    'CurveName', 'CurveType', 'Unused', 'CurveUse',
    'X1Min', 'X1Max', 'X2Min', 'X2Max',
    'C0', 'C1', 'C2', 'C3', 'C4', 'C5'
]

with open(json_template_file, 'r') as f:
    data_json = json.load(f)

grid = data_json['performance']['performance_map_cooling']['grid_variables']
keys = list(grid.keys())
values = [grid[k] for k in keys]
combinations = list(itertools.product(*values))

def _pick(df, t):
    s = df[df['CurveType'] == t]
    if s.empty:
        raise RuntimeError(f"CurveType '{t}' not found in {csv_file}")
    return s.iloc[0]

cap_f_t_row   = _pick(df_curves, 'cap-f-t')
cap_f_flow_row= _pick(df_curves, 'cap-f-ff')
eir_f_t_row   = _pick(df_curves, 'eir-f-t')
eir_f_flow_row= _pick(df_curves, 'eir-f-ff')
plf_f_plr_row = _pick(df_curves, 'plf-f-plr')

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
        P_kPa  = combo_dict["ambient_absolute_air_pressure"]

        Tdbo_C = Tdbo_K - 273.15
        WBi_C  = compute_wetbulb(Tdbi_K, RH_frac, P_kPa)

        flow_rate  = combo_dict["indoor_coil_air_mass_flow_rate"]          # kg/s
        flow_ratio = flow_rate / max(grid["indoor_coil_air_mass_flow_rate"])
        plr        = 1.0  # fixed PLR for now

        # Curves use indoor wet-bulb and outdoor dry-bulb
        gross_cap, _, power = calculate_performance(
            cap_f_t_row, cap_f_flow_row, eir_f_t_row, eir_f_flow_row, plf_f_plr_row,
            WBi_C, Tdbo_C, flow_ratio, plr,
            nominal_capacity, nominal_eir
        )

        # --- Apply stage degradation, if present ---
        comp_stage = combo_dict.get("compressor_sequence_number", 1)
        if comp_stage > 1:
            degradation_factor = 0.5
            gross_cap *= degradation_factor
            power     *= degradation_factor

        # --- Sensible capacity via ADP method ---
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

data_json['performance']['performance_map_cooling']['lookup_variables'] = lookup

Path(output_json_file).parent.mkdir(parents=True, exist_ok=True)
with open(output_json_file, 'w') as f:
    json.dump(data_json, f, indent=2)

print(f"✅ STD205 JSON updated and saved to: {output_json_file}")


# -------------------------
# STEP 3: Validate JSON
# -------------------------
base_dir = Path.cwd()
schema_file = base_dir / "RS0004.schema.json"
ashrae_file = base_dir / "ASHRAE205.schema.json"
json_file = base_dir / output_json_file

with schema_file.open() as f:
    schema = json.load(f)
with ashrae_file.open() as f:
    ashrae_schema = json.load(f)

ashrae_uri = ashrae_file.resolve().as_uri()
rs0004_uri = schema_file.resolve().as_uri()

store = {ashrae_uri: ashrae_schema, rs0004_uri: schema}
resolver = RefResolver(base_uri=rs0004_uri, referrer=schema, store=store)

with json_file.open() as f:
    data_val = json.load(f)

validator = Draft7Validator(schema, resolver=resolver)
errors = sorted(validator.iter_errors(data_val), key=lambda e: e.path)

if not errors:
    print(f"✅ {json_file.name} is valid according to {schema_file.name}")
else:
    print(f"❌ {json_file.name} has {len(errors)} validation errors:")
    for err in errors:
        path = ".".join(str(x) for x in err.path)
        print(f" - {path}: {err.message}")

# -------------------------
# STEP 4: Convert JSON to XLSX using tk205
# -------------------------
src_dir = "input"  # directory with JSON files
out_dir = "xlsx"  # output directory

tk205.translate_directory(src_dir, out_dir)
print(f"✅ Converted JSON in {src_dir} → XLSX in {out_dir}")

# -------------------------
# STEP 5: Plot validation figures (curves vs JSON table)
# -------------------------
figs_dir = Path("figs")
figs_dir.mkdir(exist_ok=True, parents=True)

# ---- 5.1: Load curves (bi-quad coefficients for eir-f-t and cap-f-t) ----
dfc = df_curves
dfc.columns = [
    "CurveName","CurveType","Unused","CurveUse",
    "X1Min","X1Max","X2Min","X2Max",
    "C0","C1","C2","C3","C4","C5"
]

def _row_to_coeffs(df, curve_type):
    row = df[df["CurveType"] == curve_type]
    if row.empty:
        raise RuntimeError(f"CurveType '{curve_type}' not found in {csv_file}")
    return row.iloc[0][["C0","C1","C2","C3","C4","C5"]].astype(float).to_numpy()

eir_coeffs = _row_to_coeffs(dfc, "eir-f-t")
cap_coeffs = _row_to_coeffs(dfc, "cap-f-t")

def bi_quad(x, y, c):
    return c[0] + c[1]*x + c[2]*x**2 + c[3]*y + c[4]*y**2 + c[5]*x*y

def bi_quad_norm(x, y, c, x_ref, y_ref):
    return bi_quad(x, y, c) / bi_quad(x_ref, y_ref, c)

# ---- 5.2: Build a performance table from the JSON we just wrote ----
with open(output_json_file, "r") as f:
    std205 = json.load(f)

grid = std205["performance"]["performance_map_cooling"]["grid_variables"]
lookup = std205["performance"]["performance_map_cooling"]["lookup_variables"]

# Recreate combinations deterministically from the JSON grid
grid_keys = list(grid.keys())
grid_vals = [grid[k] for k in grid_keys]
grid_combos = list(itertools.product(*grid_vals))

# Sanity check: combos length must match lookup arrays
n_pts = len(grid_combos)
for k in ["gross_total_capacity", "gross_power"]:
    if len(lookup[k]) != n_pts:
        raise RuntimeError(f"Lookup '{k}' length {len(lookup[k])} != grid size {n_pts}")

# Build a compact DataFrame for plotting
perf_rows = []
for i, combo in enumerate(grid_combos):
    cd = dict(zip(grid_keys, combo))
    perf_rows.append({
        "indoor_db_C":  cd["indoor_coil_entering_dry_bulb_temperature"] - 273.15,
        "outdoor_db_C": cd["outdoor_coil_entering_dry_bulb_temperature"] - 273.15,
        "capacity_W":   float(lookup["gross_total_capacity"][i]),
        "power_W":      float(lookup["gross_power"][i]),
    })
perf = pd.DataFrame(perf_rows).dropna(subset=["capacity_W","power_W"])

# ---- 5.3: Choose a common reference point and compute modifiers ----
target_in_C, target_out_C = 20.0, 30.0
# Find nearest actual table point to the desired reference
i_ref = ((perf["indoor_db_C"] - target_in_C).abs()
       + (perf["outdoor_db_C"] - target_out_C).abs()).idxmin()
cap_ref = perf.loc[i_ref, "capacity_W"]
eir_ref = perf.loc[i_ref, "power_W"] / max(perf.loc[i_ref, "capacity_W"], 1e-9)

perf["cap_mod"] = perf["capacity_W"] / max(cap_ref, 1e-9)
perf["eir_mod"] = (perf["power_W"] / perf["capacity_W"]) / max(eir_ref, 1e-9)

# ---- 5.4: Produce 1D comparisons vs outdoor temp for each indoor temp ----
outdoor_span = np.linspace(perf["outdoor_db_C"].min(), perf["outdoor_db_C"].max(), 200)

def curve_line(indoor_C, outdoor_arr, coeffs):
    return np.array([bi_quad_norm(indoor_C, o, coeffs, target_in_C, target_out_C) for o in outdoor_arr])

# EIR modifier plot
plt.figure(figsize=(7, 5))
for indoor_C in sorted(perf["indoor_db_C"].unique()):
    subset = perf[perf["indoor_db_C"] == indoor_C]
    eir_curve = curve_line(indoor_C, outdoor_span, eir_coeffs)
    plt.plot(outdoor_span, eir_curve, label=f"Curve {indoor_C:.0f}°C")
    plt.scatter(subset["outdoor_db_C"], subset["eir_mod"], marker="x", label=f"Table {indoor_C:.0f}°C")
plt.title("EIR modifier vs Outdoor Dry-Bulb")
plt.xlabel("Outdoor Dry-Bulb (°C)")
plt.ylabel("EIR Modifier (norm. to ~20°C/30°C)")
plt.legend()
eir_fig_path = figs_dir / "eir_modifier_vs_outdoor.png"
plt.tight_layout()
plt.savefig(eir_fig_path, dpi=200)
plt.show()
print(f"📈 Saved: {eir_fig_path}")

# Capacity modifier plot
plt.figure(figsize=(7, 5))
for indoor_C in sorted(perf["indoor_db_C"].unique()):
    subset = perf[perf["indoor_db_C"] == indoor_C]
    cap_curve = curve_line(indoor_C, outdoor_span, cap_coeffs)
    plt.plot(outdoor_span, cap_curve, label=f"Curve {indoor_C:.0f}°C")
    plt.scatter(subset["outdoor_db_C"], subset["cap_mod"], marker="x", label=f"Table {indoor_C:.0f}°C")
plt.title("Capacity modifier vs Outdoor Dry-Bulb")
plt.xlabel("Outdoor Dry-Bulb (°C)")
plt.ylabel("Capacity Modifier (norm. to ~20°C/30°C)")
plt.legend()
cap_fig_path = figs_dir / "capacity_modifier_vs_outdoor.png"
plt.tight_layout()
plt.savefig(cap_fig_path, dpi=200)
plt.show()
print(f"📈 Saved: {cap_fig_path}")