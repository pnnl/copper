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
"""
Generate performance curve CSVs for unitary DX equipment using the `copper` library.
Replicates the Jupyter notebook workflow (no plots, no IDF export).
"""

import os
import glob
import argparse
import copper as cp

def generate_curves(lib_path, outdir, combined_csv=None, seed=1):
    # Load copper library
    lib = cp.Library(path=lib_path)  # noqa: F841 (kept to initialize)

    # Capacity buckets (kBtu/h) and fan power (kW)
    capacities = {
        "65_to_135": 96,
        "135_to_240": 180,
        "240_to_760": 480,
        "gt760": 792,
    }
    fan_power = {
        "65_to_135": 0.524,
        "135_to_240": 1.197,
        "240_to_760": 5.243,
        "gt760": 11.190,
    }

    # ASHRAE 90.1 efficiency requirements
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
            },
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
            },
        },
    }

    os.makedirs(outdir, exist_ok=True)

    for code, req in requirements.items():
        for cap_key, ref_cap_kbtuh in capacities.items():
            if cap_key not in req["ieer"]:
                continue

            tonnage = cp.Units(value=ref_cap_kbtuh, unit="kbtu/h").conversion(new_unit="ton")

            indoor_fan_speeds = 1 if "2004" in code else 2

            dx = cp.UnitaryDirectExpansion(
                compressor_type="scroll",
                condenser_type="air",
                compressor_speed="constant",
                ref_cap_unit="ton",
                ref_net_cap=tonnage,
                full_eff=req["eer"][cap_key],
                full_eff_unit="eer",
                part_eff_ref_std="ahri_340/360",
                indoor_fan_speeds=indoor_fan_speeds,
                indoor_fan_power=fan_power[cap_key],
                indoor_fan_power_unit="kW",
            )

            if req["eer"][cap_key] is None:
                dx.full_eff = dx.ieer_to_eer(req["ieer"][cap_key])
                name = f"AC_Perf_{code}_{cap_key}_{round(dx.full_eff,2)}EER_{round(req['ieer'][cap_key],2)}IEER"
            if req["ieer"][cap_key] is None:
                agg_only = True
                name = f"AC_Perf_{code}_{cap_key}_{round(dx.full_eff,2)}EER"
            else:
                dx.part_eff = req["ieer"][cap_key]
                agg_only = False
                name = f"AC_Perf_{code}_{cap_key}_{round(dx.full_eff,2)}EER_{round(dx.part_eff,2)}IEER"

            dx.degradation_coefficient = 0.25 if "2004" in code else 0.15
            dx.add_cycling_degradation_curve(overwrite=True)

            _ = dx.generate_set_of_curves(
                method="nearest_neighbor",
                tol=0.01,
                num_nearest_neighbors=5,
                verbose=False,
                vars=["eir-f-t"],
                random_seed=seed,
                agg_only=agg_only,
            )

            dx.add_cycling_degradation_curve(overwrite=True)

            curves = cp.SetofCurves()
            curves.curves = dx.set_of_curves
            curves.eqp = dx

            limits = dx.get_ranges()
            for c in curves.curves:
                xs = limits[c.out_var]["vars_range"][0]
                c.x_min, c.x_max = xs[0], xs[1]
                if len(limits[c.out_var]["vars_range"]) > 1:
                    ys = limits[c.out_var]["vars_range"][1]
                    c.y_min, c.y_max = ys[0], ys[1]
                if "eir" in c.out_var:
                    c.out_min = 0.0
                if "plf" in c.out_var:
                    c.out_min = 0.0

            curves.export(path=outdir, fmt="csv", name=name)

    if combined_csv:
        combined_path = os.path.join(outdir, combined_csv)
        with open(combined_path, "w", encoding="utf-8") as nf:
            nf.write("name,variable,unit_type,curve_type,min_x,max_x,min_y,max_y,coeff1,coeff2,coeff3,coeff4,coeff5,coeff6\n")
            pattern = os.path.join(outdir, "AC_Perf*.csv")
            for f in glob.glob(pattern):
                with open(f, "r", encoding="utf-8") as cf:
                    for line in cf:
                        nf.write(line)
        print(f"✅ Combined CSV written: {combined_path}")

    print(f"✅ Individual CSVs written to: {os.path.abspath(outdir)}")

parser = argparse.ArgumentParser(description="Generate DX curve CSVs with copper")
parser.add_argument("--lib", default="./copper/data/unitarydirectexpansion_curves.json",
                    help="Path to copper library JSON")
parser.add_argument("--outdir", default=".",
                    help="Output directory for AC_Perf*.csv")
parser.add_argument("--combined", default="",
                    help="Optional combined CSV filename")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed")
args = parser.parse_args()

generate_curves(args.lib, args.outdir, args.combined, args.seed)


# -------------------------
# STEP 2: Populate STD205 JSON
# -------------------------

csv_file = "AC_Perf_901_2022_65_to_135_11.55EER_14.8IEER.CSV"  # generated from STEP 1
psychrolib.SetUnitSystem(psychrolib.SI)

json_template_file = "DX-Constant-Efficiency.RS0004.a205.json"
output_json_file = "input/DX_Updated_STD205_Output.json" # make the json output to be saved in input folder

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
