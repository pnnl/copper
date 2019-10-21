import random
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import time
from eppy import modeleditor
from eppy.modeleditor import IDF
from multiprocessing import Pool, cpu_count
import pickle

def wrapper(eval_curve, args):
    return eval_curve(*args)

def eval_curve(x, y, curve_type, x_min, y_min, x_max, y_max, out_min, out_max, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8, coeff9, coeff10
):
    x = min(max(x, x_min), x_max)
    y = min(max(y, y_min), y_max)
    
    if curve_type == 'bi_quad':
        out = coeff1 + coeff2 * x + coeff3 * x**2 + coeff4 * y + coeff5 * y**2 + coeff6 * x * y
        return min(max(out, out_min), out_max)
    if curve_type == 'bi_cub':
        out = coeff1 + coeff2 * x + coeff3 * x**2 + coeff4 * y + coeff5 * y**2 + coeff6 * x * y + coeff7 * x**3 + coeff8 * y**3 + coeff9 * y * x**2 + coeff10 * x * y**2
        return min(max(out, out_min), out_max)
    if curve_type == 'quad':
        out = coeff1 + coeff2 * x + coeff3 * x**2
        return min(max(out, out_min), out_max)

def conv_to_kwpton(value, unit):
    if unit == 'COP':
        return 12 / (value * 3.412)
    if unit == 'kwpton':
        return value
    if unit == 'EER':
        return 12 / value

def get_curves(curve_set):
    curves_csv_file_name = './curves.csv'
    curves_csv = open(curves_csv_file_name, 'r').read().split('\n')
    #curves_csv = curves_csv.split(',')
    curves = {}
    curves_csv_headers = [item for item in curves_csv[0].split(',')[1:]]
    for curve in curves_csv:
        curve = curve.split(',')
        if curve[0] == curve_set:
            curve_data = {}
            for idx, data in enumerate(curves_csv_headers):
                if idx > 7:
                    curve_data[data] = float(curve[idx+1])
                else:
                    curve_data[data] = curve[idx+1]
            curves[curve[7]] = curve_data
    return curves

# Returns all the info of a curve for a particular output variable
def get_curve_info(out_var, curve_set):
    #print(curve_set[out_var])
    curve_info = list(curve_set[out_var].values())[6:]
    return [curve_info[1]] + [float(i) for i in curve_info[2:]]

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
            PLR = Load # EnergyPlus consider that PLR = Load, but I wonder if it shouldn't be PLR = Load / CapOp
            EIR_f_CHW_ECT = wrapper(eval_curve, [CHW, ECT[idx]] + get_curve_info('eir-f-T', curve_set))
            EIR_f_PLR = wrapper(eval_curve, [PLR, dT] + get_curve_info('eir-f-PLR-dT', curve_set))
            EIR = EIR_ref * EIR_f_CHW_ECT * EIR_f_PLR / PLR
            kWpTon.append(EIR / 3.412141633 * 12)

        IPLV = 1 / ((0.01 / kWpTon[0]) + (0.42 / kWpTon[1]) + (0.45 / kWpTon[2]) + (0.12 / kWpTon[3]))

        return IPLV
    except:
        return 0

# Calculates kW/ton at rating conditions for a particular set of curves and kW/ton
def kwpton_calcs(kWpTon_ref, indiv, condenser_type):
    Load_ref = 1
    EIR_ref = 1 / (12 / kWpTon_ref / 3.412141633)    

    # Test conditions, based on values used by EnergyPlus
    # Might be different than actual AHRI req'ts
    Load = 1
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

    dT = ECT[0] - CHW
    Cap_f_CHW_ECT = wrapper(eval_curve, [CHW, ECT[0]] + get_curve_info('cap-f-T', indiv))
    CapOp = Load_ref * Cap_f_CHW_ECT
    PLR = Load # EnergyPlus consider that PLR = Load, but I wonder if it shouldn't be PLR = Load / CapOp
    EIR_f_CHW_ECT = wrapper(eval_curve, [CHW, ECT[0]] + get_curve_info('eir-f-T', indiv))
    EIR_f_PLR = wrapper(eval_curve, [PLR, dT] + get_curve_info('eir-f-PLR-dT', indiv))
    EIR = EIR_ref * EIR_f_CHW_ECT * EIR_f_PLR / PLR

    return EIR / 3.412141633 * 12

# Define random number between XX and XX with 4 decimal
def get_random():
    while True:
        val = float(random.randrange(-99999, 99999) / 10**(random.randint(7, 11)))#(random.randint(5, 9)))
        if val != 0:
            return val

# Individual
def individual(curve_set_origin):
    new_curve = copy.deepcopy(curve_set_origin)
    #rejec_test = False
    #while rejec_test == False:
    for var in new_curve:
        for i in range(1, 10):
            new_curve[var]['coeff' + str(i)] = new_curve[var]['coeff' + str(i)] + get_random()
    #  rejec_test = rejec_indiv(new_curve)
    return new_curve

def rejec_indiv(indiv):
    pts = 10
    ECT = np.linspace(10, 35, pts)
    CHW = 6.57
    cap_vals = []
    eir_vals = []
    unrealistic_cap_mod = False
    for i in range(len(ECT)):
        cap_val = wrapper(eval_curve, [CHW, ECT[i]] + get_curve_info('cap-f-T', indiv))
        cap_vals.append(cap_val)
        eir_val = wrapper(eval_curve, [CHW, ECT[i]] + get_curve_info('eir-f-T', indiv))
        eir_vals.append(eir_val)
        if cap_val > 1.5:
            unrealistic_cap_mod = True
    if unrealistic_cap_mod == True:
        print(1)
        return False
    elif min(cap_vals) < 0.95:
        print(2)
        return False
    elif not(monotonic(cap_vals)):
        print(3)
        return False
    elif not(monotonic(eir_vals)):
        print(4)
        return False
    elif max(eir_vals) > 1:
        print(5)
        return False
    elif monotonic(cap_vals) and strictly_decreasing(cap_vals) and sum(cap_vals) > 0 and sum(cap_vals) > pts:
        return True
    elif monotonic(eir_vals) and strictly_increasing(eir_vals) and sum(eir_vals) > 0:
        return True
    else:
        return False#False

def mutation(individual):
    new_individual = copy.deepcopy(individual)
    for var in new_individual:
        idx = random.randint(1, 10)
        new_individual[var]['coeff' + str(idx)] = new_individual[var]['coeff' + str(idx)] + get_random()
    return new_individual

# Population
def population(count, curve_set_origin):
    population = []
    for x in range(count):
        population.append(individual(curve_set_origin))
    return population

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

def get_mins(out_var, curve_set):
    xmin = get_curve_info(out_var,curve_set)[1]
    ymin = get_curve_info(out_var,curve_set)[2]
    xmax = get_curve_info(out_var,curve_set)[3]
    ymax = get_curve_info(out_var,curve_set)[4]
    return [xmin, ymin, xmax, ymax]

def fitness_min_curves(out_var, curve_set_origin, indiv):
    mins = get_mins(out_var, curve_set_origin)
    min_origin = wrapper(eval_curve, [mins[0], mins[1]] + get_curve_info(out_var, curve_set_origin))
    min_indiv = wrapper(eval_curve, [mins[0], mins[1]] + get_curve_info(out_var, indiv))
    return abs(min_origin - min_indiv)

# Fitness
def fitness(indiv, kWpton, target, condenser_type, curve_set_origin):
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
    dT = ECT[0] - CHW
    PLR = 1
       
    # EIR-f-T
    EIR_T_target = indiv['eir-f-T']['out_max'] if indiv['eir-f-T']['out_max'] < 999 else 1
    fit_EIR_T = abs(EIR_T_target - wrapper(eval_curve, [CHW, ECT[0]] + get_curve_info('eir-f-T', indiv)))
    # EIR-f-PLR
    EIR_PLR_target = indiv['eir-f-PLR-dT']['out_max'] if indiv['eir-f-PLR-dT']['out_max'] < 999 else 1
    fit_EIR_PLR = abs(EIR_PLR_target - wrapper(eval_curve, [PLR, dT] + get_curve_info('eir-f-PLR-dT', indiv)))
    # Cap-f-T
    CAP_T_target = indiv['cap-f-T']['out_max'] if indiv['cap-f-T']['out_max'] < 999 else 1
    fit_Cap_T = abs(CAP_T_target - wrapper(eval_curve, [CHW, ECT[0]] + get_curve_info('cap-f-T', indiv)))
    # Rated kW/ton
    fit_kWpton = abs(kWpton - kwpton_calcs(kWpton, indiv, condenser_type))
    # IPLV Calc
    fit_IPLV = abs(target - round(iplv_calcs(kWpton, indiv, condenser_type),4))
    
    fit_Cap_T_25 = abs(1.4 - wrapper(eval_curve, [CHW, 25] + get_curve_info('cap-f-T', indiv)))
    fit_Cap_T_15 = abs(1.5 - wrapper(eval_curve, [CHW, 15] + get_curve_info('cap-f-T', indiv)))

    return (fit_EIR_T + fit_EIR_PLR + fit_Cap_T + 1 * fit_kWpton + 3 * fit_IPLV + fit_Cap_T_25 + fit_Cap_T_15) / 9


def grade(pop, target, kWpton, condenser_type, curve_set_origin):
    summed = sum((fitness(x, kWpton, target, condenser_type, curve_set_origin) for x in pop))
    return summed / (len(pop) * 1.0)

def rejec_parent(indiv, kWpton, condenser_type):
    pts = 10
    ECT = np.linspace(10, 35, pts)
    CHW = 6.57
    cap_vals = []
    eir_vals = []
    unrealistic_cap_mod = False
    for i in range(len(ECT)):
        cap_val = wrapper(eval_curve, [CHW, ECT[i]] + get_curve_info('cap-f-T', indiv))
        cap_vals.append(cap_val)
        eir_val = wrapper(eval_curve, [CHW, ECT[i]] + get_curve_info('eir-f-T', indiv))
        eir_vals.append(eir_val)
        if cap_val > 1.3:
            unrealistic_cap_mod = True
    if unrealistic_cap_mod == True:
        return False
    elif iplv_calcs(kWpton, indiv, condenser_type) == 0:
        return False
    elif min(cap_vals) < 1:
        return False
    elif not(monotonic(cap_vals)):
        return False
    elif not(monotonic(eir_vals)):
        return False
    elif max(eir_vals) > 1:
        return False
    elif monotonic(cap_vals) and strictly_decreasing(cap_vals) and sum(cap_vals) > 0 and sum(cap_vals) > pts:
        return True
    elif monotonic(eir_vals) and strictly_increasing(eir_vals) and sum(eir_vals) > 0:
        return True
    else:
        return False

def evolve(pop, kWpton, target, curve_set_origin, condenser_type, retain=0.30, random_select=0.10, mutate=0.08):
    # Calculate fitness of each individual
    fitnesses = [ fitness(x, kWpton, target, condenser_type, curve_set_origin) for x in pop ] 
    
    # Scale the fitness scores to prevent best performers
    # from draggin the whole population to a local extremum
    # linear scaling: a + b * f
    a = max(fitnesses)
    b = min(fitnesses) / len(fitnesses)
    
    # Sort fitness scors
    graded = [ (a + fitness(x, kWpton, target, condenser_type, curve_set_origin) * b, x) for x in pop ]
    test = sorted(graded, key=lambda tup: tup[0])
    graded = [ x[1] for x in test ]
    
    # Retain best performers as parents
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    
    # Randomly add other individuals to
    # promote genetic diversity
    for ind in graded[retain_length:]:
        if random_select > random.random():
            parents.append(ind)
    
    # Mutate some individuals
    for idx, ind in enumerate(parents):
        if mutate > random.random():
            parents[idx] = mutation(ind)
    
    # Reject parents with Cap-f-T curve that is not strictly decreasing
    #cnt = 0
    #while len(parents) > 20:
    #  if not rejec_parent(parents[cnt], kWpton, condenser_type):
    #    parents.remove(parents[cnt])
    #    cnt = cnt
    #  else:
    #    cnt = cnt + 1
    #
    #if len(parents) < 20:
    #    while len(parents) < 20:
    #        parents.append(individual(curve_set_origin))
    
    # Crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = {}
            for i in range(half):
                child[male[list(male.keys())[i]]['out_var']] = male[list(male.keys())[i]]
            for i in range(half,len(male)):
                child[female[list(female.keys())[i]]['out_var']] = female[list(female.keys())[i]]
            children.append(child)        
    parents.extend(children)
    return parents

# Functions retuns the best fitness score given and its corresponding IPLV 
def find_curves_coeffs(count, curve_set_origin, target, kWpton, max_nb_generation, condenser_type, tol):
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
    dT = ECT[0] - CHW
    PLR = 1
    
    ##
    ## Try by zero-ing out all coeffs.
    ##
    
    pop = population(count, curve_set_origin)
    best_pop = pop
    fit_history = []
    iplv = 0
    kwpton_ = 0
    cnt = 0
    kwpton = kWpton
    while cnt <= max_nb_generation and not(iplv < target * (1 + tol) and iplv > target * (1 - tol) and kwpton_ < kwpton * (1 + tol) and kwpton_ > kwpton * (1 - tol)):
        pop = evolve(pop, kWpton, target, curve_set_origin, condenser_type)
        fit_history.append(grade(pop, target, kWpton, condenser_type, curve_set_origin))
        if grade(pop, target, kWpton, condenser_type, curve_set_origin) < grade(best_pop, target, kWpton, condenser_type, curve_set_origin):
            best_pop = pop
        best_indiv = best_pop[0]
        for indiv in best_pop:
            if fitness(indiv, kWpton, target, condenser_type, curve_set_origin) < fitness(best_indiv, kWpton, target, condenser_type, curve_set_origin):
                best_indiv = indiv
        iplv = round(iplv_calcs(kWpton, best_indiv, condenser_type),3)
        kwpton_ = round(kwpton_calcs(kWpton, best_indiv, condenser_type),3)
        cnt = cnt + 1
        #print("generation: {}, iplv: {}, kwpton: {}".format(cnt, round(iplv,3), round(kwpton_,3)))
    best_indiv = best_pop[0]
    for indiv in best_pop:
        if fitness(indiv, kWpton, target, condenser_type, curve_set_origin) < fitness(best_indiv, kWpton, target, condenser_type, curve_set_origin):
            best_indiv = indiv
    return [fitness(best_indiv, kWpton, target, condenser_type, curve_set_origin),
            round(iplv_calcs(kWpton, best_indiv, condenser_type),3),
            round(kwpton_calcs(kWpton, best_indiv, condenser_type),3),
            fit_history,
            round(wrapper(eval_curve, [CHW, ECT[0]] + get_curve_info('eir-f-T', best_indiv))),
            round(wrapper(eval_curve, [PLR, dT] + get_curve_info('eir-f-PLR-dT', best_indiv))),
            round(wrapper(eval_curve, [CHW, ECT[0]] + get_curve_info('cap-f-T', best_indiv))),
            best_indiv,
            cnt]

def avg_dist(mins, out_var, curve_base, curve_new):
    avg_dist_sq = 0
    for x, y in zip(np.linspace(mins[0], mins[2], 100), np.linspace(mins[1], mins[3], 100)):
        a = wrapper(eval_curve, [x, y] + get_curve_info(out_var, curve_base))
        b = wrapper(eval_curve, [x, y] + get_curve_info(out_var, curve_new))
        avg_dist_sq = avg_dist_sq + (a - b)**2
    return (avg_dist_sq**0.5) / 100
    
def get_curve_out(mins, out_var, curve):
    curve_out = []
    for x, y in zip(np.linspace(mins[0], mins[2], 100), np.linspace(mins[1], mins[3], 100)):
        curve_out.append(wrapper(eval_curve, [x, y] + get_curve_info(out_var, curve)))
    return curve_out

def gen_algo_n_select(tup):
    iteration = tup[0]
    nb_curves = tup[1]
    iplv_target = 0.48
    full_load_kWpton = 0.540
    max_generation = 2000
    tolerance = 0.002
    initial_population = 100
    base_curve_name = 'PNNL_AC_2010_PA'
    condenser_type = 'air_cooled'
    tolerance_set = 0.005

    iteration_curves = []
    best_curves = []
    c = 0
    while len(best_curves) < nb_curves:
        ga_results = find_curves_coeffs(initial_population, 
                                        get_curves(base_curve_name), 
                                        iplv_target, 
                                        full_load_kWpton, 
                                        max_generation, 
                                        condenser_type, 
                                        tolerance)
        if rejec_indiv:
          print("Iteration #{}, curve #{}, generated: {} generations.".format(iteration + 1, c + 1, ga_results[8]))
          iteration_curves.append(ga_results[7])
          if len(iteration_curves) >= nb_curves:
            graded = [(curve_score(base_curve_name, x), x) for x in iteration_curves]
            graded_sorted = sorted(graded, key=lambda tup: tup[0])
            #print(graded_sorted)
            x_best = graded_sorted[:nb_curves]
            print(np.std([x[0] for x in x_best]))
            if np.std([x[0] for x in x_best]) < tolerance_set:
              best_curves = [x[1] for x in x_best]
        c = c + 1
    return best_curves
    
def curve_score(base_curve_name, curve):
    avg_dist_inst = avg_dist(get_mins('cap-f-T', curve), 'cap-f-T', get_curves(base_curve_name), curve) + \
    avg_dist(get_mins('eir-f-T', curve), 'eir-f-T', get_curves(base_curve_name), curve) + \
    avg_dist(get_mins('eir-f-PLR-dT', curve), 'eir-f-PLR-dT', get_curves(base_curve_name), curve)
    return avg_dist_inst
    
def curve_score_(base_curve_name, curve):
    avg_dist_inst = avg_dist(get_mins('cap-f-T', curve), 'cap-f-T', base_curve_name, curve) + \
    avg_dist(get_mins('eir-f-T', curve), 'eir-f-T', base_curve_name, curve) + \
    avg_dist(get_mins('eir-f-PLR-dT', curve), 'eir-f-PLR-dT', base_curve_name, curve)
    return avg_dist_inst

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ ==  '__main__': 
    iterations = 10
    time1 = time.time()
    for i, nb_c in enumerate([5]):#[5, 10, 15, 20]:
      p=Pool(processes = min(cpu_count() - 1, iterations))
      ga_curves = list(p.map(gen_algo_n_select,[(i, nb_c) for i in range(iterations)]))
      save_obj(ga_curves, 'ga_curves_n_{}_Cap'.format(nb_c,str(i)))
    time2 = time.time()
    print("Finished in {} s.".format(time2 - time1))