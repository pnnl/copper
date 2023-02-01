"""
generator.py
====================================
The generator module of Copper deals with generating performance curves.
It currently uses a genetic algorithm to find sets of curves that match user equipment definitions.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy
import random
import logging

from copper.library import *
from copper.curves import *

location = os.path.dirname(os.path.realpath(__file__))
chiller_lib = os.path.join(location, "lib", "chiller_curves.json")


class Generator:
    def __init__(
        self,
        equipment,
        method="best_match",
        pop_size=100,
        tol=0.005,
        max_gen=300,
        vars="",
        sFac=0.5,
        retain=0.2,
        random_select=0.1,
        mutate=0.95,
        bounds=(6, 10),
        base_curves=[],
        random_seed=None,
        num_nearest_neighbors=10,
        max_restart=None,
    ):
        self.equipment = equipment
        self.method = method
        self.pop_size = pop_size
        self.tol = tol
        self.max_gen = max_gen
        self.max_restart = max_restart
        self.vars = vars
        self.sFac = sFac
        self.retain = retain
        self.random_select = random_select
        self.mutate = mutate
        self.bounds = bounds
        self.base_curves = base_curves

        self.num_nearest_neighbors = num_nearest_neighbors

        if isinstance(random_seed, int):
            random.seed(random_seed)

    def generate_set_of_curves(self, verbose=False):
        """Generate set of curves using genetic algorithm.

        :param str verbose: Output results at each generation.
        :return: Set of curves
        :rtype: SetofCurves

        """
        self.target = self.equipment.part_eff
        self.full_eff = self.equipment.full_eff
        self.target_alt = self.equipment.part_eff_alt
        self.full_eff_alt = self.equipment.full_eff_alt

        if len(self.base_curves) == 0:
            if self.method == "best_match":
                lib, filters = self.equipment.get_lib_and_filters()
                self.base_curves = [lib.find_base_curves(filters, self.equipment)]
            else:
                seed_curves = self.equipment.get_seed_curves()  # get seed curves
                ranges, misc_attr = self.equipment.ranges, self.equipment.misc_attr
                if self.method == "nearest_neighbor":
                    self.base_curve = seed_curves.get_aggregated_set_of_curves(
                        ranges=ranges,
                        misc_attr=misc_attr,
                        method="NN-weighted-average",
                        N=self.num_nearest_neighbors,
                    )
                    self.base_curves = [self.base_curve]
                    self.df, _ = seed_curves.nearest_neighbor_sort(
                        target_attr=misc_attr, N=self.num_nearest_neighbors
                    )
                elif self.method == "weighted_average":
                    self.base_curve = seed_curves.get_aggregated_set_of_curves(
                        ranges=ranges, misc_attr=misc_attr, method="weighted-average"
                    )
                    self.base_curves = [self.base_curve]
                    self.df, _ = seed_curves.nearest_neighbor_sort(
                        target_attr=misc_attr
                    )
                else:
                    self.base_curves = None
                    self.df = None

        self.set_of_base_curves = self.base_curves[0]
        self.set_of_base_curves.eqp = self.equipment
        self.set_of_base_curves.eqp.set_of_curves = self.set_of_base_curves.curves
        self.base_curves_data = {}
        for curve in self.set_of_base_curves.curves:
            self.base_curves_data[
                curve.out_var
            ] = self.set_of_base_curves.get_data_for_plotting(curve, False)

        # Run generator
        res = self.run_ga(curves=self.base_curves, verbose=verbose)

        if res is None:
            return
        else:
            return self.equipment.set_of_curves

    def run_ga(self, curves, verbose=False):
        """Run genetic algorithm.

        :param copper.curves.SetofCurves curves: Initial set of curves to be modified by the algorithm
        :param str verbose: Output results at each generation.
        :return: Final population of sets of curves
        :rtype: list

        """

        self.pop = []
        gen = 0
        restart = 0
        self.equipment.curves = curves

        while not self.is_target_met():
            self.pop = self.generate_population(curves)
            while gen <= self.max_gen and not self.is_target_met():
                self.evolve_population(self.pop)
                gen += 1
                if verbose:
                    if self.target_alt > 0:
                        part_rating_alt = round(
                            self.equipment.calc_rated_eff(
                                eff_type="part",
                                unit=self.equipment.part_eff_unit_alt,
                                alt=True,
                            ),
                            4,
                        )
                    else:
                        part_rating_alt = "n/a"
                    if self.full_eff_alt > 0:
                        full_rating_alt = round(
                            self.equipment.calc_rated_eff(
                                eff_type="full",
                                unit=self.equipment.full_eff_unit_alt,
                                alt=True,
                            ),
                            4,
                        )
                    else:
                        full_rating_alt = "n/a"
                    logging.info(
                        "GEN: {}, IPLV: {}, {}: {} IPLV-alt: {}, {}-alt: {}".format(
                            gen,
                            round(
                                self.equipment.calc_rated_eff(
                                    eff_type="part", unit=self.equipment.part_eff_unit
                                ),
                                4,
                            ),
                            self.equipment.part_eff_unit.upper(),
                            round(
                                self.equipment.calc_rated_eff(
                                    eff_type="full", unit=self.equipment.full_eff_unit
                                ),
                                4,
                            ),
                            self.equipment.part_eff_unit.upper(),
                            part_rating_alt,
                            full_rating_alt,
                        )
                    )
                max_gen = gen

            if not self.is_target_met():
                if self.max_restart is not None:
                    if restart < self.max_restart:
                        logging.warning(
                            f"Target not met after {self.max_gen} generations; Restarting the generator."
                        )
                        gen = 0
                        restart += 1

                        logging.info(
                            "GEN: {}, IPLV: {}, KW/TON: {}".format(
                                gen,
                                round(
                                    self.equipment.calc_rated_eff(
                                        eff_type="part",
                                        unit=self.equipment.part_eff_unit,
                                    ),
                                    2,
                                ),
                                round(
                                    self.equipment.calc_rated_eff(
                                        eff_type="full",
                                        unit=self.equipment.full_eff_unit,
                                    ),
                                    2,
                                ),
                            )
                        )
                    else:
                        logging.critical(
                            f"Target not met after {self.max_restart} restart; No solution was found."
                        )
                        return

        logging.info("Target met after {} generations.".format(gen))
        return self.pop

    def is_target_met(self):
        """Check if the objective of the optimization through the algorithm have been met.

        :return: Verification result
        :rtype: bool

        """
        if self.equipment.type == "chiller":
            if self.equipment.set_of_curves != "":
                part_rating = self.equipment.calc_rated_eff(
                    eff_type="part", unit=self.equipment.part_eff_unit
                )
                full_rating = self.equipment.calc_rated_eff(
                    eff_type="full", unit=self.equipment.full_eff_unit
                )
                if self.target_alt > 0:
                    part_rating_alt = self.equipment.calc_rated_eff(
                        eff_type="part", unit=self.equipment.part_eff_unit_alt, alt=True
                    )
                else:
                    part_rating_alt = 0
                if self.full_eff_alt > 0:
                    full_rating_alt = self.equipment.calc_rated_eff(
                        eff_type="full", unit=self.equipment.full_eff_unit_alt, alt=True
                    )
                else:
                    full_rating_alt = 0
                cap_rating = 0
                if "cap-f-t" in self.vars:
                    for c in self.equipment.set_of_curves:
                        # set_of_curves
                        if "cap" in c.out_var:
                            cap_rating += abs(1 - c.get_out_reference(self.equipment))
            else:
                return False
        else:
            raise ValueError("This type of equipment has not yet been implemented.")

        if (
            (part_rating <= self.target * (1 + self.tol))
            and (part_rating >= self.target * (1 - self.tol))
            and (part_rating_alt <= self.target_alt * (1 + self.tol))
            and (part_rating_alt >= self.target_alt * (1 - self.tol))
            and (full_rating <= self.full_eff * (1 + self.tol))
            and (full_rating >= self.full_eff * (1 - self.tol))
            and (full_rating_alt <= self.full_eff_alt * (1 + self.tol))
            and (full_rating_alt >= self.full_eff_alt * (1 - self.tol))
            and (cap_rating <= self.tol)
            and (cap_rating >= -self.tol)
            # and self.check_gradients()
        ):
            return True
        else:
            return False

    def check_gradients(self):
        """Check if the objective of the gradient of the three curves are monotonic and have the sign we expect.

        :return: Verification result
        :rtype: bool

        """
        if self.equipment.type == "chiller":
            if self.equipment.set_of_curves != "":
                grad_list = []
                for c in self.equipment.set_of_curves:
                    if (
                        c.out_var == "eir-f-t"
                        or c.out_var == "eir-f-plr"
                        or c.out_var == "eir-f-plr-dt"
                    ):
                        sign_val = +1
                    elif c.out_var == "cap-f-t":
                        sign_val = -1
                    else:
                        raise ValueError("this curve output has not been implemented")

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

    def compute_grad(self, x, y, sign_val, threshold=1e-5):
        """Check, for a single curve, if the gradient has the sign we expect. called by check_gradients.

        :return: Verification result
        :rtype: bool

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

        :param copper.curves.SetofCurves curves: Initial set of curves to be modified by the algorithm
        :return: Verification result
        :rtype: bool

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

        :param copper.curves.SetofCurves curves: Initial set of curves to be modified by the algorithm
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

    def calc_fit(self, pop):
        """Calculate fitness of each individual in a population.

        :param list pop: Population
        :return: Fitness score of each individual
        :rtype: list

        """
        # Determine and normalized part load efficiency fitness
        part_load_fitnesses = [
            self.determine_part_load_fitness(curves) for curves in pop
        ]
        part_load_fitnesses = [
            score / max(part_load_fitnesses) for score in part_load_fitnesses
        ]

        # Determine and normalized full load efficiency fitness
        full_load_fitnesses = [
            self.determine_full_load_fitness(curves) for curves in pop
        ]
        full_load_fitnesses = [
            score / max(full_load_fitnesses) for score in full_load_fitnesses
        ]

        # Determine and normalized "normalization" fitness
        if len(self.vars) > 1:
            normalization_fitnesses = [
                self.determine_normalization_fitness(curves) for curves in pop
            ]
            if max(normalization_fitnesses) > 0:
                normalization_fitnesses = [
                    score / max(normalization_fitnesses)
                    for score in normalization_fitnesses
                ]
        else:
            normalization_fitnesses = [0.0 for curves in pop]

        # Sum fitness scores
        overall_fitnesses = [
            sum(x)
            for x in zip(
                *[part_load_fitnesses, full_load_fitnesses, normalization_fitnesses]
            )
        ]
        return overall_fitnesses

    def fitness_scale_grading(self, pop, scaling=True):
        """Calculate fitness, scale, and grade for a population.

        :param list pop: Population of individual, i.e. list of curves
        :param bool scaling: Linearly scale fitness scores
        :return: List sets of curves graded by fitness scores
        :rtype: list

        """
        # Determine fitness
        overall_fitnesses = self.calc_fit(pop)

        # Scaling
        pop_scaled = self.scale_fitnesses(overall_fitnesses, pop, scaling)

        # Grading
        pop_graded = self.grade_population(pop_scaled)

        return pop_graded

    def evolve_population(self, pop):
        """Evolve population to create a new generation.

        :param list pop: Population of individual, i.e. list of curves

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

    def determine_part_load_fitness(self, set_of_curves):
        """Determine the part load fitness of a set of curves

        :param copper.curves.SetofCurves set_of_curves: Set of curves
        :return: Fitness score
        :rtype: float

        """
        # Temporary assign curve to equipment
        self.equipment.set_of_curves = set_of_curves.curves
        part_eff_score = abs(
            self.equipment.calc_rated_eff(
                eff_type="part", unit=self.equipment.part_eff_unit
            )
            - self.target
        )
        if self.target_alt > 0:
            part_eff_score += abs(
                self.equipment.calc_rated_eff(
                    eff_type="part", unit=self.equipment.part_eff_unit_alt, alt=True
                )
                - self.target_alt
            )
        return part_eff_score

    def determine_full_load_fitness(self, set_of_curves):
        """Determine the full load fitness of a set of curves

        :param copper.curves.SetofCurves set_of_curves: Set of curves
        :return: Fitness score
        :rtype: float

        """
        # Temporary assign curve to equipment
        self.equipment.set_of_curves = set_of_curves.curves
        full_eff_score = abs(
            self.equipment.calc_rated_eff(
                eff_type="full", unit=self.equipment.full_eff_unit
            )
            - self.equipment.full_eff
        )
        if self.equipment.full_eff_alt > 0:
            full_eff_score += abs(
                self.equipment.calc_rated_eff(
                    eff_type="full", unit=self.equipment.full_eff_unit_alt, alt=True
                )
                - self.equipment.full_eff_alt
            )
        return full_eff_score

    def determine_normalization_fitness(self, set_of_curves):
        """Determine the normalization fitness of a set of curves (e.g. are they normalized at rated/reference conditions?)

        :param copper.curves.SetofCurves set_of_curves: Set of curves
        :return: Fitness score
        :rtype: float

        """
        # Temporary assign curve to equipment
        self.equipment.set_of_curves = set_of_curves.curves
        curve_normal_score = 0
        for c in set_of_curves.curves:
            if c.out_var in self.vars:
                curve_normal_score += abs(1 - c.get_out_reference(self.equipment))
        return curve_normal_score

    def scale_fitnesses(self, fitnesses, pop, scaling=True):
        """Scale the fitness scores to prevent best performers from dragging the whole population to a local extremum.

        :param list fitnesses: List of fitness for each set of curves
        :param list pop: List of sets of curves
        :param bool scaling: Specifies whether of not to linearly scale the fitnesses
        :return: List of tuples representing the fitness of a set of curves and the set of curves
        :rtype: list

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
            (a * score + b, set_of_curves)
            for score, set_of_curves in zip(*[fitnesses, pop])
        ]
        return pop_scaled

    def grade_population(self, pop_scaled):
        """Grade population.

        :param list pop_scaled: List of tuples representing the fitness of a set of curves and the set of curves
        :return: List of set of curves graded from the best to the worst
        :rtype: list

        """
        pop_sorted = sorted(pop_scaled, key=lambda tup: tup[0])
        pop_graded = [item[1] for item in pop_sorted]
        return pop_graded

    def perform_mutation(self, individual):
        """Mutate individual.

        :param copper.curves.SetofCurves individual: Set of curves
        :return: Modified indivudal
        :rtype: SetofCurves

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

        :param list parents: List of best performing individuals of the generation

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
                child = SetofCurves()
                child.eqp = self.equipment
                curves = []
                # male and female curves are structured the same way
                for _, c in enumerate(male.curves):
                    # Copy as male
                    n_child_curves = copy.deepcopy(c)
                    if c.out_var in self.vars or len(self.vars) == 0:
                        if c.type == "quad":
                            positions = [[1], [2, 3]]  # cst  # x^?
                        elif c.type == "cubic":
                            positions = [[1], [2, 3, 4]]  # cst  # x^?
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
