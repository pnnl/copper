import json


def add_curves_CSV_to_JSON(action_type="w"):
    """
    Convert curve defined in a CSV file to a JSON
    """
    csv_c = open("../fixtures/curves.csv", "r")
    csv_c.readline()

    new_db_entry = {}

    for line in csv_c:
        litems = line.split(",")
        if not "reformulated" in litems[1].lower():
            name = litems[0]
            model = litems[1]
            ref_cap = float(litems[2])
            ref_cop = float(litems[3])
            comp_type = litems[4]
            cond_type = litems[5]
            speed = litems[6]
            out_var = litems[7]
            ctype = litems[8]
            x_min = float(litems[9])
            y_min = float(litems[10])
            x_max = float(litems[11])
            y_max = float(litems[12])
            out_min = float(litems[13])
            out_max = float(litems[14])
            coeffs = []
            for i in range(15, 25):
                coeffs.append(float(litems[i].replace("\n", "")))
            if not name in list(new_db_entry.keys()):
                new_db_entry[name] = {}
            curve = new_db_entry[name]
            curve["eqp_type"] = "chiler"
            curve["compressor_type"] = comp_type
            curve["condenser_type"] = cond_type
            curve["sim_engine"] = "energyplus"
            curve["algorithm"] = (
                "ect&lwt" if model.lower() == "chiller:electric:eir" else "ect&lct"
            )
            curve["speed"] = speed
            curve["ref_cap"] = ref_cap
            # curve["ref_cap_unit"] = "ton"
            curve["full_eff"] = ref_cop
            curve["full_eff_unit"] = "COP"
            # curve["part_eff"] = "TDB"
            # curve["part_eff_unit"] = "COP"

            if not "curves" in list(curve.keys()):
                curve["curves"] = {}
            curve["curves"][out_var] = {}
            curve["curves"][out_var]["curve_type"] = ctype
            curve["curves"][out_var]["x_min"] = x_min
            curve["curves"][out_var]["x_max"] = x_max
            curve["curves"][out_var]["y_min"] = y_min
            curve["curves"][out_var]["y_max"] = y_max
            curve["curves"][out_var]["out_min"] = out_min
            curve["curves"][out_var]["out_max"] = out_max
            curve["curves"][out_var]["coeffs"] = coeffs
    with open("../fixtures/curves_from_csv.json", "w", encoding="utf-8") as f:
        json.dump(new_db_entry, f, ensure_ascii=False, indent=4)


add_curves_CSV_to_JSON()
