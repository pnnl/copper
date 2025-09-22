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

# -------------------------
# STEP 1: Generate performance CSV
# -------------------------
def cooling_capacity_curve(T_outdoor, T_indoor, flow_rate):
    base_capacity = 35000  # Btu/h nominal
    adjustment = (
        -200 * (T_outdoor - 95) 
        + 150 * (75 - T_indoor) 
        + 50 * (flow_rate - 400)
    )
    return base_capacity + adjustment

outdoor_temps = np.arange(70, 115, 5)
indoor_temps = [72, 75, 78]
flow_rates = [350, 400, 450]

data = []
for t_out in outdoor_temps:
    for t_in in indoor_temps:
        for flow in flow_rates:
            cap = cooling_capacity_curve(t_out, t_in, flow)
            data.append((t_out, t_in, flow, cap))

df = pd.DataFrame(data, columns=["OutdoorDB", "IndoorDB", "CFM", "Capacity_BtuH"])

# Save to CSV
csv_file = "unitary_dx_capacity_curves.csv"
df.to_csv(csv_file, index=False)
print(f"✅ Performance CSV saved to {csv_file}")

# Plot example
plt.figure(figsize=(8, 6))
for t_in in indoor_temps:
    subset = df[df["IndoorDB"] == t_in]
    plt.plot(subset["OutdoorDB"], subset["Capacity_BtuH"], marker="o", label=f"Indoor {t_in}F")
plt.xlabel("Outdoor Dry Bulb (F)")
plt.ylabel("Capacity (Btu/h)")
plt.title("Cooling Capacity vs Outdoor Temperature")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# STEP 2: Populate STD205 JSON
# -------------------------
psychrolib.SetUnitSystem(psychrolib.SI)

json_template_file = "DX-Constant-Efficiency.RS0004.a205.json"
output_json_file = "DX_Updated_STD205_Output.json"

# Nominal values
nominal_capacity = 232057  # W
nominal_eer = 9.2
nominal_eir = 1 / nominal_eer
nominal_SHR = 0.7

def compute_wetbulb(Tdb_K, RH_frac, pressure_kPa):
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
    return gross_capacity, 0.0, power

# Load curves
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

cap_f_t_row = df_curves[df_curves['CurveType'] == 'cap-f-t'].iloc[0]
cap_f_flow_row = df_curves[df_curves['CurveType'] == 'cap-f-ff'].iloc[0]
eir_f_t_row = df_curves[df_curves['CurveType'] == 'eir-f-t'].iloc[0]
eir_f_flow_row = df_curves[df_curves['CurveType'] == 'eir-f-ff'].iloc[0]
plf_f_plr_row = df_curves[df_curves['CurveType'] == 'plf-f-plr'].iloc[0]

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
        plr = 1.0

        gross_cap, _, power = calculate_performance(
            cap_f_t_row, cap_f_flow_row, eir_f_t_row, eir_f_flow_row, plf_f_plr_row,
            WBi_C, Tdbo_C, flow_ratio, plr, nominal_capacity, nominal_eir
        )

        lookup["gross_total_capacity"].append(gross_cap)
        lookup["gross_sensible_capacity"].append(gross_cap * nominal_SHR)  # simplified
        lookup["gross_power"].append(power)
        lookup["operation_state"].append("NORMAL")

    except Exception as e:
        print(f"Skipping point {combo_dict} due to error: {e}")
        continue

data_json['performance']['performance_map_cooling']['lookup_variables'] = lookup

with open(output_json_file, 'w') as f:
    json.dump(data_json, f, indent=2)

print(f"✅ STD205 JSON updated and saved to: {output_json_file}")

# -------------------------
# STEP 3: Validate JSON
# -------------------------
base_dir = Path.cwd()
schema_file = base_dir / "RS0004.schema.json"
ashrae_file = base_dir / "ASHRAE205.schema.json"
json_file   = base_dir / output_json_file

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
src_dir = "input"   # directory with JSON files
out_dir = "xlsx"    # output directory

tk205.translate_directory(src_dir, out_dir)
print(f"✅ Converted JSON in {src_dir} → XLSX in {out_dir}")
