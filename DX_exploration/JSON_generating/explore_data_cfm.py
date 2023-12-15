import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json


class DXPerfMap:

    def __init__(
            self,
            xlsx_file,
            sheet_ind,
            rated_flow_rate=3000, # change
            comp_sequences=1,
    ):

        self.xlsx_file = xlsx_file
        self.sheet_ind = sheet_ind
        self.rated_flow_rate = rated_flow_rate
        self.comp_sequences = comp_sequences

        self.df = self.load_xlsx()
        self.Te_db, self.Te_wb = self.load_AHRI_conditions()
        self.df_rated, self.rated_capacity, self.rated_sensible_capacity, self.df_rated_T, self.df_rated_flow = self.get_rated_conditions(self.df)

    def load_xlsx(self):
        df = pd.read_excel(io=self.xlsx_file, sheet_name=self.sheet_ind)
        df["indoor_coil_entering_dry_bulb_temperature"] = self.convert_F_to_C(
            df["indoor_coil_entering_dry_bulb_temperature"])
        df["outdoor_coil_entering_dry_bulb_temperature"] = self.convert_F_to_C(
            df["outdoor_coil_entering_dry_bulb_temperature"])
        df["indoor_coil_entering_wet_bulb_temperature"] = self.convert_F_to_C(
            df["indoor_coil_entering_wet_bulb_temperature"])
        return df

    @staticmethod
    
    def convert_F_to_C(F):
        return (F-32) * 5/9

    ## have changed to DX Equipment's
    def load_AHRI_conditions(self):
        Te_db = 26.7 # 80 F
        Te_wb = 19.4 # 67 F
        return Te_db, Te_wb

    def get_rated_conditions(self, df):
        T_db = df["indoor_coil_entering_dry_bulb_temperature"].values
        T_wb = df["indoor_coil_entering_wet_bulb_temperature"].values
        dist = (T_db - self.Te_db)**2 + (T_wb - self.Te_wb)**2
        # check the location of minimum distance
        min_idx = np.argmin(dist)
        df_rated = self.df.iloc[min_idx]
        rated_capacity = df_rated["gross_total_capacity"]
        rated_sensible_capacity = df_rated["gross_sensible_capacity"]
        # get rated capacity
        self.df["cap_f_t"] = self.df["gross_total_capacity"]/rated_capacity
        self.df["ff"] = self.df["indoor_coil_air_mass_flow_rate"]/self.rated_flow_rate
        # check the compressor stage
        stage_len = len(np.unique(self.df["compressor_sequence_number"].values))

        ##Get the dataframe for cap-f-t
        if self.comp_sequences == 1 or stage_len == 1: # compressor stage = 1 or 2 only
            df_const_T = self.df[
                (np.abs(self.df["indoor_coil_entering_dry_bulb_temperature"] - self.Te_db) < 1.0) &
                (np.abs(self.df["indoor_coil_entering_wet_bulb_temperature"] - self.Te_wb) < 1.0)
            ]

            #get the dataframe
            df_rated_flow = self.df[(np.abs(self.df["indoor_coil_air_mass_flow_rate"] - self.rated_flow_rate) < 100)]
        else:
            df_const_T = dict()
            df_rated_flow = dict()
            for num_comp in range(self.comp_sequences):
                _df = self.df.loc[self.df["compressor_sequence_number"] == (num_comp+1)]
                df_const_T[f"comp_seq_{num_comp+1}"] =  _df[
                (np.abs(_df["indoor_coil_entering_dry_bulb_temperature"] - self.Te_db) < 1.0) &
                (np.abs(_df["indoor_coil_entering_wet_bulb_temperature"] - self.Te_wb) < 1.0)
             ]
                df_rated_flow[f"comp_seq_{num_comp+1}"] = _df[
                    (np.abs(_df["indoor_coil_air_mass_flow_rate"] - self.rated_flow_rate) < 100)]
        return df_rated, rated_capacity, rated_sensible_capacity, df_const_T, df_rated_flow



class CurveFit:

    def __init__(self, xlsx_file, x, targets, fun="bi-quad", y=None):
        self.xlsx_file = xlsx_file
        self.x = x
        self.y = y
        self.targets = targets
        self.function = fun

        if self.function == "bi-quad":
            self.X = self.f_biquad()
        elif self.function == "cubic":
            self.X = self.f_cubic()
        else:
            self.X = None

        assert self.X is not None
        self.coeffs = None

    def f_biquad(self):
        assert self.y is not None
        X = np.vstack(
            (
                np.ones(self.x.shape[0], ),
                self.x,
                self.x ** 2,
                self.y,
                self.y ** 2,
                np.multiply(self.x, self.y),
            )
        )

        return np.transpose(X)

    def f_cubic(self):
        X = np.vstack(
            (
                np.ones(self.x.shape[0], ),
                self.x,
                self.x ** 2,
                self.x ** 3,
            )
        )

        return np.transpose(X)

    def fit(self):
        coeffs = np.linalg.lstsq(self.X, self.targets)
        self.coeffs = coeffs[0]
        return self.coeffs

    def plot(self, file_str, sheet_ind='Table 1', label="Cap-f-t"):
        """
        Method to compare the predicted and
        """
        # Find the index of "./data/" and ".xlsx"
        xlsx_file = self.xlsx_file
        start_index = xlsx_file.find("./data/DX_Equipment_Data_Collection_") + len("./data/DX_Equipment_Data_Collection_")
        end_index = xlsx_file.find(".xlsx")
        # Extract the information between "./data/" and ".xlsx"
        extracted_info = xlsx_file[start_index:end_index]
        fig_dir = "./figures/" + extracted_info + "_" + sheet_ind + "/"

        # fig_dir = "./figures/D_" + sheet_ind + "/"

        #create directory
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        assert self.coeffs is not None
        predictions = np.matmul(self.X, self.coeffs)
        print(f"The Pearson coefficient is : {np.corrcoef(self.targets, predictions)}")

        #plot and get the R^2 metric
        plt.plot(self.targets, predictions, 'ko')
        plt.plot(self.targets, self.targets, 'k-')
        plt.xlabel(f"{label} (Actual)", fontsize=16)
        plt.ylabel(f"{label} (Predictions)", fontsize=16)
        plt.savefig(os.path.join(fig_dir, f"{file_str}_R2_{np.corrcoef(self.targets, predictions)[0, 1]}_fit.png"))

        return None

def plot(xlsx_file, sheet_ind="Table 1"):
    # Find the index of "./data/" and ".xlsx"
    start_index = xlsx_file.find("./data/DX_Equipment_Data_Collection_") + len("./data/DX_Equipment_Data_Collection_")
    end_index = xlsx_file.find(".xlsx")
    # Extract the information between "./data/" and ".xlsx"
    extracted_info = xlsx_file[start_index:end_index]
    
    fig_dir = "./figures/" + extracted_info + "_" + sheet_ind + "/"
    df = pd.read_excel(io=xlsx_file, sheet_name=sheet_ind)
    #df.columns = df.index
    #Define grid variables
    x_cols = [
        "outdoor_coil_entering_dry_bulb_temperature",
        "indoor_coil_entering_wet_bulb_temperature",
        "indoor_coil_entering_dry_bulb_temperature",
        "indoor_coil_air_mass_flow_rate",
        "compressor_sequence_number"
    ]

    #define lookup variables
    y_cols = [
        "gross_total_capacity",
        "gross_sensible_capacity",
        "gross_power"
    ]

    #create directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for yc in y_cols:
        for xc in x_cols:
            plt.plot(df[xc], df[yc], 'ko')
            plt.xlabel(xc, fontsize=16)
            plt.ylabel(yc, fontsize=16)
            plt.savefig(os.path.join(fig_dir, f"filename_{yc}_vs_{xc}.png"))
            plt.close()

    return None


# run all
def run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate):
    DX = DXPerfMap(xlsx_file=xlsx_file, sheet_ind=sheet_ind,
                    rated_flow_rate=flow_rate,comp_sequences=num_comp_sequences)
    plot(xlsx_file, sheet_ind)

    if xlsx_file[46] == "L" and num_comp_sequences == 1:
        compressor_speed = "variable"
    else:
        compressor_speed = "constant"
    equip_data = {"eqp_type": "unitary_dx",
                  'equip_label': xlsx_file[46:-5]+sheet_ind[-1],
                  'num_comp_sequences': num_comp_sequences,
                  'ref_flow_rate': flow_rate,
                  'ref_Te_db': DX.Te_db,
                  'ref_Te_wb': DX.Te_wb,
                  'ref_cap': DX.rated_capacity,
                  'ref_sensible_cap': DX.rated_sensible_capacity,
                  'ref_cap_unit': 'Btu/h',
                  "compressor_type": "scroll",
                  "condenser_type": "air",
                  "compressor_speed": compressor_speed,
                  "sim_engine": "energyplus"
                  # To Do ? :
                  # 'OAT': 'outdoor_coil_entering_dry_bulb_temperature'
                  # 'gross_sensible_capacity'
                  # 'gross_power'
                  # 'indoor_coil_air_mass_flow_rate'
                  # 'supply_fan_power'
                  }
    #test ff
    curve_set_data = []
    for nc in range(num_comp_sequences):
        if num_comp_sequences == 1:
            df_rated_flow = DX.df_rated_flow
        else:
            df_rated_flow = DX.df_rated_flow[f"comp_seq_{nc+1}"]
        TotCapTFit = CurveFit(xlsx_file,
            x=df_rated_flow["outdoor_coil_entering_dry_bulb_temperature"].values,
            targets=df_rated_flow["cap_f_t"].values,
            y=df_rated_flow["indoor_coil_entering_wet_bulb_temperature"].values
        )
        #print("Curve fitting for Case I")
        coef_case1 = TotCapTFit.fit()
        case1_data = {"num_comp_seq": nc,
        "cap-f-t": {
            "out_var": "cap-f-t",
            "type": "bi_quad",
        #    "ref_evap_fluid_flow": null,
        #    "ref_cond_fluid_flow": null,
        #    "ref_lwt": 6.67,
        #    "ref_ect": 35.0,
        #    "ref_lct": null,
            "units": "si",
            "x_min": df_rated_flow["outdoor_coil_entering_dry_bulb_temperature"].values.min(),
            "y_min": df_rated_flow["indoor_coil_entering_wet_bulb_temperature"].values.min(),
            "x_max": df_rated_flow["outdoor_coil_entering_dry_bulb_temperature"].values.max(),
            "y_max": df_rated_flow["indoor_coil_entering_wet_bulb_temperature"].values.max(),
            "out_min": df_rated_flow["cap_f_t"].values.min(),
            "out_max": df_rated_flow["cap_f_t"].values.max(),
            "coeff1": coef_case1[0],
            "coeff2": coef_case1[1],
            "coeff3": coef_case1[2],
            "coeff4": coef_case1[3],
            "coeff5": coef_case1[4],
            "coeff6": coef_case1[5],
            "coeff7": 0.0,
            "coeff8": 0.0,
            "coeff9": 0.0,
            "coeff10": 0.0
            }
        }
        curve_set_data.append(case1_data)
        print(f"coefficients for Case I: {coef_case1}")
        TotCapTFit.plot(file_str=f"TotCapFlowTempFit_seq_{nc+1}", sheet_ind=DX.sheet_ind)

        # ####New Curve fit objects
        if num_comp_sequences == 1:
            df_rated_T = DX.df_rated_T
        else:
            df_rated_T = DX.df_rated_T[f"comp_seq_{nc+1}"]
        TotCapFlowFit = CurveFit(xlsx_file,
            x=df_rated_T["ff"].values,
            targets=df_rated_T["cap_f_t"].values,
            fun="cubic"
        )
        # print("Curve Fitting for Case II")
        coef_case2 = TotCapFlowFit.fit()
        case2_data = {"num_comp_seq": nc,
        "cap-f-ff": {
            "out_var": "cap-f-ff",
            "type": "cubic",
        #    "ref_evap_fluid_flow": null,
        #    "ref_cond_fluid_flow": null,
        #    "ref_lwt": 6.67,
        #    "ref_ect": 35.0,
        #    "ref_lct": null,
            "units": "si",
            "x_min": df_rated_T["ff"].values.min(),
            "x_max": df_rated_T["ff"].values.max(),
            "out_min": df_rated_T["cap_f_t"].values.min(),
            "out_max": df_rated_T["cap_f_t"].values.max(),
            "coeff1": coef_case2[0],
            "coeff2": coef_case2[1],
            "coeff3": coef_case2[2],
            "coeff4": coef_case2[3],
            "coeff5": 0.0,
            "coeff6": 0.0,
            "coeff7": 0.0,
            "coeff8": 0.0,
            "coeff9": 0.0,
            "coeff10": 0.0
            }
        }
        curve_set_data.append(case2_data)
        print(f"coefficients for Case II: {coef_case2}")
        TotCapFlowFit.plot(file_str=f"TotCapFlowFit_seq_{nc+1}", sheet_ind=DX.sheet_ind, label="Cap-f-ff")
    print(f"{xlsx_file}{sheet_ind} ran succefully.")

    # Create a new dictionary with the list under one key
    set_of_curves = {"set_of_curves": curve_set_data}
    equip_data.update(set_of_curves)
    # Convert the updated dictionary to a JSON string
    json_data = json.dumps(equip_data, indent=4)
    return json_data

if __name__ == "__main__":

    ### 1. DX_Equipment_Data_Collection_D.xlsx
    xlsx_file = "./data/processed/DX_Equipment_Data_Collection_D.xlsx"
    json_list = []
    ## manually run to check each table to get 'num_comp_sequences' and 'flow_rate'
        # Table 1
    sheet_ind = "Table 1"
    num_comp_sequences = 1 # 1 compressor
    flow_rate = 2100
    j1d = run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)
    json_list.append(j1d)

        # Table 2
    sheet_ind = "Table 2"
    num_comp_sequences = 1 # 2 compressors
    flow_rate = 3000
    j2d = run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)
    json_list.append(j2d)

        # Table 3
    sheet_ind = "Table 3"
    num_comp_sequences = 1 # 2 compressors
    flow_rate = 3400
    j3d = run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)
    json_list.append(j3d)

        # Table 4
    sheet_ind = "Table 4"
    num_comp_sequences = 1 # 2 compressors
    flow_rate = 4000
    j4d = run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)
    json_list.append(j4d)

        # Table 5
    sheet_ind = "Table 5"
    num_comp_sequences = 1 # 2 compressors
    flow_rate = 5000
    j5d = run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)
    json_list.append(j5d)

        # Table 6
    sheet_ind = "Table 6"
    num_comp_sequences = 1 # 3 compressors
    flow_rate = 5010
    j6d = run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)
    json_list.append(j6d)

        # Table 7
    sheet_ind = "Table 7"
    num_comp_sequences = 1 # 4 compressors
    flow_rate = 6475
    j7d = run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)
    json_list.append(j7d)

    #     # Table 8
    # sheet_ind = "Table 8"
    # num_comp_sequences = 1 # 5 compressors
    # flow_rate = 8250
    # j8d =run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)

# ### 2. DX_Equipment_Data_Collection_L.xlsx
#     xlsx_file = "./data/processed/DX_Equipment_Data_Collection_L.xlsx"

#     ## manually run to check each table to get 'num_comp_sequences' and 'flow_rate'
#         # Table 1
#     sheet_ind = "Table 1"
#     num_comp_sequences = 1 # 1 compressor
#     flow_rate = 2100
#     run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)

### 3. DX_Equipment_Data_Collection_C_SM.xlsx
    # xlsx_file = "./data/processed/DX_Equipment_Data_Collection_C_SM.xlsx"

    # ## manually run to check each table to get 'num_comp_sequences' and 'flow_rate'
    #     # Table 1
    # sheet_ind = "Table 1"
    # num_comp_sequences = 1 # 1 compressor
    # flow_rate = 2100
    # run_all(xlsx_file,sheet_ind,num_comp_sequences,flow_rate)

    # Specify the filename
    filename = 'DX_cfm.json'

    # Write the list of dictionaries to a JSON file
    with open(filename, 'w') as file:
        json.dump(json_list, file, indent=4)
