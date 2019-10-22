def iplv_calcs(kWpTon_ref, curve_set, condenser_type):
    try:
        Load_ref = 1
        EIR_ref = 1 / (12 / kWpTon_ref / 3.412141633)    

        # Test conditions, based on values used by EnergyPlus
        # Might be different than actual AHRI req'ts
        Loads = [1, 0.75, 0.5, 0.25]
        if condenser_type == 'air_cooled':
            CHW = 6.67
            ECT = [3 + 32 * Loads[0],
                3 + 32 * Loads[1],
                3 + 32 * Loads[2],
                13]
        elif condenser_type == 'water_cooled':
            CHW = 6.67
            ECT = [8 + 22 * Loads[0],
                8 + 22 * Loads[1],
                19,
                19]

        kWpTon = []

        for idx, Load in enumerate(Loads):
            dT = ECT[idx] - CHW
            Cap_f_CHW_ECT = wrapper(eval_curve, [CHW, ECT[idx]] + get_curve_info('cap-f-T', curve_set))
            CapOp = Load_ref * Cap_f_CHW_ECT
            PLR = Load / CapOp # EnergyPlus consider that PLR = Load, but I wonder if it shouldn't be PLR = Load / CapOp
            EIR_f_CHW_ECT = wrapper(eval_curve, [CHW, ECT[idx]] + get_curve_info('eir-f-T', curve_set))
            EIR_f_PLR = wrapper(eval_curve, [PLR, dT] + get_curve_info('eir-f-PLR-dT', curve_set))
            EIR = EIR_ref * EIR_f_CHW_ECT * EIR_f_PLR / PLR
            kWpTon.append(EIR / 3.412141633 * 12)

        IPLV = 1 / ((0.01 / kWpTon[0]) + (0.42 / kWpTon[1]) + (0.45 / kWpTon[2]) + (0.12 / kWpTon[3]))

        return IPLV
    except:
        return 0