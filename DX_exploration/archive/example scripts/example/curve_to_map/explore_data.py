import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from WetBulb import WetBulb

class DXPerfMap:

    def __init__(
            self,
            xlsx_file,
            rated_flow_rate=1.11,
            comp_sequences=1,
    ):

        self.xlsx_file = xlsx_file
        self.rated_flow_rate = rated_flow_rate
        self.comp_sequences = comp_sequences

        self.df = self.load_xlsx()
        self.Te_db, self.Te_wb = self.load_AHRI_conditions()
        self.df_rated, self.rated_capacity, self.df_rated_T, self.df_rated_flow = self.get_rated_conditions(self.df)

    def load_xlsx(self):
        df = pd.read_excel(io=self.xlsx_file, sheet_name="performance_map_cooling", skiprows=[0, 1, 3])
        df["indoor_coil_entering_dry_bulb_temperature"] = self.convert_K_to_C(
            df["indoor_coil_entering_dry_bulb_temperature"])
        df["outdoor_coil_entering_dry_bulb_temperature"] = self.convert_K_to_C(
            df["outdoor_coil_entering_dry_bulb_temperature"])

        T_wb, _, _ = WetBulb(
            TemperatureC=df["indoor_coil_entering_dry_bulb_temperature"].values,
            Pressure=df["ambient_absolute_air_pressure"].values,
            Humidity=100*df["indoor_coil_entering_relative_humidity"].values, #relative humidity is in 100%
            HumidityMode=1
        )
        df["indoor_coil_entering_wet_bulb_temperature"] = T_wb
        return df

    @staticmethod
    def convert_K_to_C(K):
        return K - 273.0

    def load_AHRI_conditions(self):
        Te_db = 26.7
        Te_wb = 19.4
        return Te_db, Te_wb

    def get_rated_conditions(self, df):
        T_db = df["outdoor_coil_entering_dry_bulb_temperature"].values
        T_wb = df["indoor_coil_entering_wet_bulb_temperature"].values
        dist = (T_db - self.Te_db)**2 + (T_wb - self.Te_wb)**2
        #check the location of minimum distance
        min_idx = np.argmin(dist)
        df_rated = self.df.iloc[min_idx]
        rated_capacity = df_rated["gross_total_capacity"]
        #get rated capacity
        self.df["cap_f_t"] = self.df["gross_total_capacity"]/rated_capacity
        self.df["ff"] = self.df["indoor_coil_air_mass_flow_rate"]/self.rated_flow_rate

        ##Get the dataframe for cap-f-t
        if self.comp_sequences == 1:
            df_const_T = self.df[
                (np.abs(self.df["outdoor_coil_entering_dry_bulb_temperature"] - self.Te_db) < 1.0) &
                (np.abs(self.df["indoor_coil_entering_wet_bulb_temperature"] - self.Te_wb) < 1.0)
            ]

            #get the dataframe
            df_rated_flow = self.df[(np.abs(self.df["indoor_coil_air_mass_flow_rate"] - self.rated_flow_rate) < 0.1)]
        else:
            df_const_T = dict()
            df_rated_flow = dict()
            for num_comp in range(self.comp_sequences):
                _df = self.df.loc[self.df["compressor_sequence_number"] == (num_comp+1)]
                df_const_T[f"comp_seq_{num_comp+1}"] =  _df[
                (np.abs(_df["outdoor_coil_entering_dry_bulb_temperature"] - self.Te_db) < 1.0) &
                (np.abs(_df["indoor_coil_entering_wet_bulb_temperature"] - self.Te_wb) < 1.0)
             ]
                df_rated_flow[f"comp_seq_{num_comp+1}"] = _df[
                    (np.abs(_df["indoor_coil_air_mass_flow_rate"] - self.rated_flow_rate) < 0.1)]
        return df_rated, rated_capacity, df_const_T, df_rated_flow



class CurveFit:

    def __init__(self, x, targets, fun="bi-quad", y=None):
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

    def plot(self, file_str, fig_dir="figures", label="Cap-f-t"):
        """
        Method to compare the predicted and
        """
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

def plot(xlsx_file, filename, fig_dir="./figures/"):

    df = pd.read_excel(io=xlsx_file, sheet_name="performance_map_cooling", skiprows=[0, 1, 3])
    #df.columns = df.index
    #Define grid variables
    x_cols = [
        "outdoor_coil_entering_dry_bulb_temperature",
        "indoor_coil_entering_relative_humidity",
        "indoor_coil_entering_dry_bulb_temperature",
        "indoor_coil_air_mass_flow_rate",
        "compressor_sequence_number",
        "ambient_absolute_air_pressure"
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

def test_ff(df, fig_dir="./figures/"):
    plt.plot(df["ff"], df["cap_f_t"], 'ko')
    plt.savefig(os.path.join(fig_dir, "cap-f-t_v_ff.png"))
    plt.close()

    return None
##create a helper function
def compute_Twb(T_db, RH, pressure):
    print(f"T_db: {T_db}, RH: {RH}, pressure: {pressure}")
    Hum_mode = 1
    T_wb, _, _ =  WetBulb(T_db,pressure,RH*100,Hum_mode)
    return S[5]


if __name__ == "__main__":
    xlsx_file = "./data/HPDM.RS0004.a205.xlsx"
    num_comp_sequences = 2
    DX = DXPerfMap(xlsx_file=xlsx_file, rated_flow_rate=1.11,comp_sequences=num_comp_sequences)
    
    #test ff
    for nc in range(num_comp_sequences):
        df_rated_flow = DX.df_rated_flow[f"comp_seq_{nc+1}"]
        TotCapTFit = CurveFit(
            x=df_rated_flow["outdoor_coil_entering_dry_bulb_temperature"].values,
            targets=df_rated_flow["cap_f_t"].values,
            y=df_rated_flow["indoor_coil_entering_wet_bulb_temperature"].values
        )
        #print("Curve fitting for Case I")
        TotCapTFit.fit()
        TotCapTFit.plot(file_str=f"TotCapFlowTempFit_seq_{nc+1}")

        # ####New Curve fit objects
        df_rated_T = DX.df_rated_T[f"comp_seq_{nc+1}"]
        TotCapFlowFit = CurveFit(
            x=df_rated_T["ff"].values,
            targets=df_rated_T["cap_f_t"].values,
            fun="cubic"
        )
        # print("Curve Fitting for Case II")
        TotCapFlowFit.fit()
        TotCapFlowFit.plot(file_str=f"TotCapFlowFit_seq_{nc+1}", label="Cap-f-ff")
