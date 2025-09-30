# DX Performance Mapping Generator for STD 205

This project provides tools to generate and validate **ASHRAE Standard 205-compliant performance maps** for DX (Direct Expansion) air conditioning units using curve-based performance data.  
The workflow is fully implemented in a single script: **`main.py`**.

---

## 📌 Features

* Generates performance curves using the **copper** library (ASHRAE 90.1 code-based EER/IEER requirements).
* Saves curves into individual **CSV files** (e.g., `AC_Perf_*.csv`) and an optional combined CSV.
* Computes cooling performance over a grid of conditions defined in a template JSON.
* Supports various curve types: **linear, quadratic, cubic, bi-quadratic**.
* Uses **PsychroLib** for accurate psychrometric calculations.
* Estimates sensible cooling capacity from total capacity (simplified SHR approach).
* Saves computed results into a structured JSON format compliant with ASHRAE Standard 205.
* Validates generated JSON against **RS0004** and **ASHRAE205** schemas.
* Converts JSON results to **Excel (XLSX)** format using [Toolkit 205](https://github.com/open205/toolkit-205).

---

## 📂 File Structure

* `main.py`  
  → Main script that performs the **entire workflow**:
  1. Generate performance curves (CSV)
  2. Populate STD205 JSON template with results
  3. Validate JSON against ASHRAE 205 schema
  4. Convert JSON → XLSX

* `input/`  
  → JSON files for conversion (e.g., `AC_Perf_901_2022_65_to_135_11.55EER_14.8IEER.json`).

* `xlsx/`  
  → Output folder where converted XLSX files will be saved.

* `AC_Perf_*.csv`  
  → Example CSV file generated from the copper-based curve generation.

* `DX-Constant-Efficiency.RS0004.a205.json`  
  → Template JSON defining the variable grid and structure for performance mapping.

* `RS0004.schema.json`, `ASHRAE205.schema.json`  
  → Schemas used for validation.

---

## 📥 Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib psychrolib jsonschema tk205
````

For JSON → XLSX conversion using Toolkit 205, follow setup at [Toolkit 205](https://github.com/open205/toolkit-205), then build schemas:

```bash
poetry run doit build_schema
```

---

## 🧠 Workflow

1. **Generate Curves**
   `main.py` uses the **copper** library to generate code-compliant DX performance curves (EER/IEER).
   Outputs: CSVs (`AC_Perf_*.csv`) saved in the working directory.

2. **Map to JSON**
   `main.py` reads a selected CSV (e.g., `AC_Perf_901_2022_65_to_135_11.55EER_14.8IEER.csv`), applies modifier curves, computes capacity & power, and fills the **STD205 JSON template**.

3. **Validation**
   The JSON output is validated against the **RS0004** and **ASHRAE205** schemas.

4. **Conversion** (optional)
   Using **tk205**, the script converts valid JSON into Excel `.xlsx` format (stored in `xlsx/`).

---

## ⚙️ Configuration

Key parameters inside `main.py`:

```python
csv_file = "AC_Perf_901_2022_65_to_135_11.55EER_14.8IEER.csv"  # input curve CSV
json_template_file = "DX-Constant-Efficiency.RS0004.a205.json"
output_json_file = "input/DX_Updated_STD205_Output.json"

nominal_capacity = 232057  # W
nominal_eer = 9.2
nominal_SHR = 0.7
```

---

## 🧪 Example Usage

Run the full workflow:

```bash
python main.py
```

Example outputs:

```
✅ Individual CSVs written to ./ 
✅ Combined CSV written: ieer_specific_curves.csv
✅ STD205 JSON updated and saved to: input/DX_Updated_STD205_Output.json
✅ DX_Updated_STD205_Output.json is valid according to RS0004.schema.json
✅ Converted JSON in input → XLSX in xlsx
```

---

## ⚠️ Notes

* Script assumes **PLR = 1.0** (part-load ratio fixed).
* Sensible capacity estimation is simplified (constant SHR).
* Curve types must be one of: `bi_quad`, `cubic`, `quadratic`, `linear`.
* Validation requires **RS0004.schema.json** and **ASHRAE205.schema.json** in the working directory.
