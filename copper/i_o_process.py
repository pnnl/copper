from copper.units import *


def output(ref, ref_unit, load, ect, idx, lwt, plr, cap_op, kwpton):
    # Convert the reference capacity to tons if necessary
    cap_ton = ref
    if ref_unit != "ton":
        cap_ton = Units(ref, ref_unit).conversion("ton")

    # Generate the part load report
    part_report = f"""At {str(round(load * 100.0, 0)).replace('.0', '')}% load and AHRI rated conditions:
    - Entering condenser temperature: {round(ect[idx], 2)},
    - Leaving chiller temperature: {round(lwt, 2)},
    - Part load ratio: {round(plr, 2)},
    - Operating capacity: {round(cap_op * cap_ton, 2)} ton,
    - Power: {round(kwpton * cap_op * cap_ton, 2)} kW,
    - Efficiency: {round(kwpton, 3)} kW/ton
    """

    # Return the generated report
    return part_report
