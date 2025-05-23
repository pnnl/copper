import pickle
import copper as cp
import sys
import os
import matplotlib.pyplot as plt


def save_obj(obj, name):
    with open("./res/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def plot_lib(lib_path, rating_std="ahri_340/360"):
    """
    Function to generate
    """
    DXLibrary = cp.Library(path=lib_path, rating_std=rating_std)

    #define the set of filters
    filters = [
        ("eqp_type", "UnitaryDirectExpansion"),
        ("compressor_type", "scroll"),
        ("compressor_speed", "constant/variable"),
        ("full_eff_ref_std", rating_std),
        ("condenser_type", "air"),
        ("sim_engine", "energyplus")
    ]

    #Find sets of curves
    _sets_of_curves = DXLibrary.find_set_of_curves_from_lib(filters=filters, part_eff_flag=True)

    print(f"set_of_curves: {_sets_of_curves}")
    #plot the set_of_curves
    out_var = ["eir-f-t", "cap-f-t", "eir-f-ff", "cap-f-ff"]
    #Set up axes
    fig, axes = plt.subplots(ncols=4, figsize=(10, 4))
    for sc in _sets_of_curves:
        sc.plot(out_var=out_var, axes=axes, norm=False, alpha=0.1)

    #Pair the sets of curves with Equipment object
    plt.show()
    return None



if __name__ == "__main__":
    location = os.path.dirname(os.path.realpath(__file__))
    dx_lib = os.path.join(location, "./copper/data/multi_stage.json")
    #print(dx_lib)
    plot_lib(lib_path=dx_lib)