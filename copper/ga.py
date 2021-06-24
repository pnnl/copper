import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy, random

from copper.library import *
from copper.curves import *


class GA:
    def __init__(
        self,
        equipment,
        method="typical",
        pop_size=100,
        tol=0.02,
        max_gen=15000,
        vars="",
        sFac=0.5,
        retain=0.2,
        random_select=0.1,
        mutate=0.95,
        bounds=(6, 10),
        base_curves=[],
    ):
        self.equipment = equipment
        self.method = method
        self.pop_size = pop_size
        self.tol = tol
        self.max_gen = max_gen
        self.vars = vars
        self.sFac = sFac
        self.retain = retain
        self.random_select = random_select
        self.mutate = mutate
        self.bounds = bounds
        self.base_curves = base_curves

    def generate_set_of_curves(self):
        """Generate set of curves using genetic algorithm.

        :return: Set of curves
        :rtype: SetofCurves()

        """
        self.target = self.equipment.part_eff
        self.full_eff = self.equipment.full_eff

        # Convert target if different than kw/ton
        if self.equipment.part_eff_unit != "kw/ton":
            target_c = Units(self.target, self.equipment.part_eff_unit)
            self.target = target_c.conversion("kw/ton")

        # Convert target if different than kw/ton
        if self.equipment.full_eff_unit != "kw/ton":
            full_eff_c = Units(self.equipment.full_eff, self.equipment.full_eff_unit)
            self.full_eff = full_eff_c.conversion("kw/ton")

        if len(self.base_curves) == 0:
            if self.equipment.type == "chiller":
                # TODO: implement other methods
                if self.method == "typical":
                    lib = Library(path="./fixtures/typical_curves.json")
                elif self.method == "best_match":
                    lib = Library(path="./fixtures/chiller_curves.json")

                # Define chiller properties
                filters = [
                    ("eqp_type", self.equipment.type),
                    ("comp_type", self.equipment.compressor_type),
                    ("cond_type", self.equipment.condenser_type),
                    ("comp_speed", self.equipment.compressor_speed),
                    ("sim_engine", self.equipment.sim_engine),
                    ("model", self.equipment.model),
                ]
            else:
                raise ValueError("This type of equipment has not yet been implemented.")

            # Find typical curves from library
            # Only one equipment should be returned
            if self.method == "typical":
                self.base_curves = lib.find_set_of_curvess_from_lib(filters)
            elif self.method == "best_match":
                self.base_curves = [lib.find_base_curves(filters, self.equipment)]

        self.set_of_base_curves = self.base_curves[0]
        self.base_curves_data = {}
        for curve in self.set_of_base_curves.curves:
            self.base_curves_data[
                curve.out_var
            ] = self.set_of_base_curves.get_data_for_plotting(curve, False)

        # Run GA
        self.run_ga(curves=self.base_curves)
        return self.equipment.set_of_curves

    def run_ga(self, curves):
        """Run genetic algorithm.

            :param SetofCurves() curves: Initial set of curves to be modified by the algorithm
            :return: Final population of sets of curves
            :rtype: list()

        ]"""
        self.pop = self.generate_population(curves)
        gen = 0
        self.equipment.curves = curves
        while gen <= self.max_gen and not self.is_target_met():
            self.evolve_population(self.pop)
            gen += 1
            #For debugging
            #print("GEN: {}, IPLV: {}, KW/TON: {}".format(gen, round(self.equipment.calc_eff(eff_type="iplv"),4), round(self.equipment.calc_eff(eff_type="kwpton"),4)))
        print("Curve coefficients calculated in {} generations.".format(gen))
        return self.pop

    def is_target_met(self):
        """Check if the objective of the optimization through the algorithm have been met.

        :return: Verification result
        :rtype: boolean

        """
        if self.equipment.type == "chiller":
            if self.equipment.set_of_curves != "":
                part_rating = self.equipment.calc_eff(eff_type="iplv")
                full_rating = self.equipment.calc_eff(eff_type="kwpton")
                cap_rating = 0
                if "cap-f-t" in self.vars:
                    for (
                        c
                    ) in self.equipment.set_of_curves:  # list of objects  # c in curves
                        # set_of_curves
                        if "cap" in c.out_var:
                            cap_rating += abs(1 - c.get_out_reference())
            else:
                return False
        else:
            raise ValueError("This type of equipment has not yet been implemented.")


        if (
            (part_rating < self.target * (1 + self.tol))
            and (part_rating > self.target * (1 - self.tol))
            and (full_rating < self.full_eff * (1 + self.tol))
            and (full_rating > self.full_eff * (1 - self.tol))
            and (cap_rating < self.tol)
            and (cap_rating > -self.tol)
            and self.check_gradients()
        ):
            return True
        else:
            return False

    def check_gradients(self):

        """Check if the objective of the gradient of the three curves are monotonic and have the sign we expect.

        :return: Verification result
        :rtype: boolean

        """
        if self.equipment.type == "chiller":
            if self.equipment.set_of_curves != "":

                grad_list = []
                for c in self.equipment.set_of_curves:
                    if c.out_var == "eir-f-t" or c.out_var == "eir-f-plr":
                        sign_val = +1
                    elif c.out_var == "cap-f-t":
                        sign_val = -1
                    else:
                        raise ValueError("this curve output has not been implemented")

                    #check if c_out var is in self.vars
                    if c.out_var in self.vars:
                        x, y = self.set_of_base_curves.get_data_for_plotting(c, False)
                        grad_list.append(self.compute_grad(x=x, y=y, sign_val=sign_val))
                if np.all(np.asarray(grad_list)):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def compute_grad(self, x, y, sign_val, threshold=0.15):

        """Check, for a single curve, if the gradient has the sign we expect. called by check_gradients.

        :return: Verification result
        :rtype: boolean

        """
        grad = np.around(
            np.gradient(y, x), 2
        )  # add a small number to get rid of very small negative values
        grad[
            np.abs(grad) <= threshold
        ] = 0  # making sure that small gradients are set to zero to avoid
    
        sign = np.sign(grad)
        if np.all(np.asarray(y) == 0):  # all values are false
            return False
        elif np.all(
            sign != -sign_val
        ):  # include 0 and +1/-1 gradients. but not gradients of the opposite sign
            return True
        else:
            return False

    def generate_population(self, curves):
        """Generate population of sets of curves.

        :param SetofCurves() curves: Initial set of curves to be modified by the algorithm
        :return: Verification result
        :rtype: boolean

        """
        pop = []
        for _ in range(self.pop_size):
            pop.append(self.individual(curves))
        return pop

    def get_random(self):
        """Generate random number between two bounds.

        :return: Randomly generated value
        :rtype: float

        """
        while True:
            val = float(
                random.randrange(-99999, 99999)
                / 10 ** (random.randint(self.bounds[0], self.bounds[1]))
            )
            if val != 0:
                return val

    def individual(self, curves):
        """Create new individual.

        :param SetofCurves() curves: Initial set of curves to be modified by the algorithm
        :return: New set of curves randomly modified
        :rtype: SetofCurves

        """
        new_curves = copy.deepcopy(curves[0])
        for curve in new_curves.curves:
            if len(self.vars) == 0 or curve.out_var in self.vars:
                for idx in range(1, 11):
                    try:
                        # TODO: screening criteria
                        setattr(
                            curve,
                            "coeff{}".format(idx),
                            getattr(curve, "coeff{}".format(idx)) + self.get_random(),
                        )
                    except:
                        pass
        return new_curves

    def fitness_scale_grading(self, pop, scaling=True):
        """Calculate fitness, scale, and grade for a population.

        :param list() pop: Population of individual, i.e. list of curves
        :param boolean scaling: Linearly scale fitness scores
        :return: List sets of curves graded by fitness scores
        :rtype: list()

        """
        # Intial fitness calcs
        fitnesses = [self.determine_fitness(curves) for curves in pop]

        # Scaling
        pop_scaled = self.scale_fitnesses(fitnesses, pop, scaling)

        # Grading
        pop_graded = self.grade_population(pop_scaled)

        return pop_graded

    def evolve_population(self, pop):
        """Evolve population to create a new generation.

        :param list() pop: Population of individual, i.e. list of curves

        """
        # Fitness, Scaling, Grading
        pop_graded = self.fitness_scale_grading(pop)

        # Retain best performers as parents
        retain_length = int(len(pop_graded) * self.retain)
        parents = pop_graded[:retain_length]

        # Randomly add other individuals to
        # promote genetic diversity
        for individual in pop_graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Mutate some individuals
        for idx, individual in enumerate(parents):
            if self.mutate > random.random():
                parents[idx] = self.perform_mutation(individual)

        # Crossover parents to create children
        self.perform_crossover(parents)
        self.identify_best_performer()

    def determine_fitness(self, set_of_curves):
        """Compute fitness score of an individual.

        :param SetofCurves() set_of_curves: Set of curves
        :return: Fitness score
        :rtype: float

        """
        # Temporary assign curve to equipment
        self.equipment.set_of_curves = set_of_curves.curves

        # Compute normalization score
        # and RSME with base curve
        # TODO: Try PCM
        # TODO: Try Frechet distance
        # TODO: Area between two curves
        # TODO: Dynamic Time Warping distance
        curve_normal_score = 0
        rsme = 0
        for c in set_of_curves.curves:
            if c.out_var in self.vars:
                curve_normal_score += abs(1 - c.get_out_reference())
                x, y = set_of_curves.get_data_for_plotting(c, False)
                base_x, base_y = self.base_curves_data[c.out_var]
                rsme += np.sqrt(((np.array(y) - np.array(base_y)) ** 2).mean())

        if self.equipment.type == "chiller":
            iplv_score = abs(self.equipment.calc_eff(eff_type="iplv") - self.target)
            full_eff_score = abs(
                self.equipment.calc_eff(eff_type="kwpton") - self.equipment.full_eff
            )
            iplv_weight = 1
            eff_weight = 1
            curve_normal_score_weight = 1
            rsme_weight = 0.5

            fit_score = (
                iplv_score * iplv_weight
                + full_eff_score * eff_weight
                + curve_normal_score * curve_normal_score_weight
                + rsme * rsme_weight
            ) / (
                iplv_weight
                + eff_weight
                + curve_normal_score_weight * len(set_of_curves.curves)
                + rsme_weight
            )
        else:
            raise ValueError("This type of equipment has not yet been implemented.")

        return fit_score

    def scale_fitnesses(self, fitnesses, pop, scaling=True):
        """Scale the fitness scores to prevent best performers from dragging the whole population to a local extremum.

        :param list() fitnesses: List of fitness for each set of curves
        :param list() pop: List of sets of curves
        :param boolean scaling: Specifies whether of not to linearly scale the fitnesses
        :return: List of tuples representing the fitness of a set of curves and the set of curves
        :rtype: list(tuple())

        """
        # linear scaling: a + b * f
        # Based on "Scaling in Genetic Algorithms", Sushil J. Louis
        # https://www.cse.unr.edu/~sushil/class/gas/notes/scaling/index.html
        if scaling:
            max_f = max(fitnesses)
            min_f = min(fitnesses)
            avg_f = sum(fitnesses) / len(fitnesses)
            if min_f > (self.sFac * avg_f - max_f) / (self.sFac - 1.0):
                d = max_f - avg_f
                if d == 0:
                    a = 1
                    b = 0
                else:
                    a = (self.sFac - 1.0) * avg_f
                    b = avg_f * (max_f - (self.sFac * avg_f)) / d
            else:
                d = avg_f - min_f
                if d == 0:
                    a = 1
                    b = 0
                else:
                    a = avg_f / d
                    b = -min_f * avg_f
        else:
            a = 1
            b = 0

        pop_scaled = [
            (a * self.determine_fitness(set_of_curves) + b, set_of_curves)
            for set_of_curves in pop
        ]
        return pop_scaled

    def grade_population(self, pop_scaled):
        """Grade population.

        :param list(tuple()) pop_scaled: List of tuples representing the fitness of a set of curves and the set of curves
        :return: List of set of curves graded from the best to the worst
        :rtype: list(SetofCurves())

        """
        pop_sorted = sorted(pop_scaled, key=lambda tup: tup[0])
        pop_graded = [item[1] for item in pop_sorted]
        return pop_graded

    def perform_mutation(self, individual):
        """Mutate individual.

        :param SetofCurves() individual: Set of curves
        :return: Modified indivudal
        :rtype: SetofCurves()

        """
        new_individual = copy.deepcopy(individual)
        for curve in new_individual.curves:
            if len(self.vars) == 0 or curve.out_var in self.vars:
                idx = random.randint(1, curve.nb_coeffs())
                setattr(
                    curve,
                    "coeff{}".format(idx),
                    getattr(curve, "coeff{}".format(idx)) + self.get_random(),
                )
        return new_individual

    def perform_crossover(self, parents):
        """Crossover best individuals.

        :param list() parents: List of best performing individuals of the generation

        """
        parents_length = len(parents)
        desired_length = len(self.pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)
            # Can't crossover with the same element
            if male != female:
                male = parents[male]
                female = parents[female]
                child = SetofCurves(eqp_type=self.equipment.type)
                curves = []
                # male and female curves are structured the same way
                for _, c in enumerate(male.curves):
                    # Copy as male
                    n_child_curves = copy.deepcopy(c)
                    if c.out_var in self.vars or len(self.vars) == 0:
                        if c.type == "quad":
                            positions = [[1], [2, 3]]  # cst  # x^?
                        elif c.type == "bi_quad":
                            positions = [
                                [1],  # cst
                                [2, 3],  # x^?
                                [4, 5],  # y^?
                                [6],
                            ]  # x*y
                        elif c.type == "bi_cub":
                            positions = [
                                [1],  # cst
                                [2, 3, 7],  # x^?
                                [4, 5, 8],  # y^?
                                [6],  # x*y
                                [9],  # x^2*y
                                [10],
                            ]  # x*y^2
                        else:
                            raise ValueError("Type of curve not yet implemented.")
                        couple = ["male", copy.deepcopy(female)]
                        cnt = 0
                        for p in positions:
                            # Alternate between male and female
                            cnt = (cnt + 1) % 2
                            if couple[cnt] != "male":
                                # sub_position
                                for s_p in p:
                                    setattr(
                                        n_child_curves,
                                        "coeff{}".format(s_p),
                                        getattr(n_child_curves, "coeff{}".format(s_p)),
                                    )
                    curves.append(n_child_curves)
                child.curves = curves
                children.append(child)
        parents.extend(children)
        self.pop = parents

    def identify_best_performer(self):
        """Assign the best individual to the equipment."""
        # Re-compute fitness, scaling and grading
        pop_graded = self.fitness_scale_grading(self.pop, scaling=True)

        # Assign the equipment with the best fitness
        self.equipment.set_of_curves = pop_graded[0].curves
