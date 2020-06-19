import json


def curve_csv_to_json(csv_path):
    """
    Convert curve sets defined in a CSV file using a predefined format to a JSON file
    """
    csv_c = open(csv_path, "r")

    json_c = {}
    for i, line in enumerate(csv_c):
        litems = line.replace("\n", "").split(",")
        if i == 0:
            headers = litems
        else:
            c_spec = False
            for j, h in enumerate(headers):
                if litems[j] == "":
                    litems[j] = None
                if j == 0:
                    # TODO: add equipment as argument to the function
                    json_c["eqp_type"] = "chiller"
                    json_c[litems[0]] = {}
                if h == "out_var":
                    c_spec = True
                    json_c[litems[0]]["curves"] = {}
                    out_var = litems[j]
                if not c_spec:
                    try:
                        json_c[litems[0]][h] = float(litems[j])
                    except:
                        json_c[litems[0]][h] = litems[j]
                else:
                    if not out_var in json_c[litems[0]]["curves"].keys():
                        json_c[litems[0]]["curves"][out_var] = {}
                    try:
                        json_c[litems[0]]["curves"][out_var][h] = float(litems[j])
                    except:
                        json_c[litems[0]]["curves"][out_var][h] = litems[j]
                if h == "coeff10":
                    c_spec = False

    with open(csv_path.replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(json_c, f, ensure_ascii=False, indent=4)
    return True
