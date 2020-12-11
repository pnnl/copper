from unittest import TestCase

import copper as cp
import numpy as np
import time, sys, io, pickle


class TestBenchmark(TestCase):
    def test_benchmark(self):
        chlr = cp.Chiller(
            ref_cap=300,
            ref_cap_unit="tons",
            full_eff=0.650,
            full_eff_unit="kw/ton",
            part_eff=0.48,
            part_eff_unit="kw/ton",
            sim_engine="energyplus",
            model="ect_lwt",
            compressor_type="centrifugal",
            condenser_type="water",
            compressor_speed="constant",
        )

        times = []
        nb_gens = []
        setsofcurves = []
        iterations = 20

        for _ in range(iterations):
            # Initialise time counter
            start_time = time.process_time()

            # Store initial stdout
            stdout_ = sys.stdout
            sys.stdout = io.StringIO()

            # Generate a set of performance curves for targeted equipment
            chlr.generate_set_of_curves(
                vars=["eir-f-t", "cap-f-t", "eir-f-plr"], method="typical"
            )

            # Extract number of generation needed to reach goal
            output = sys.stdout.getvalue()
            nb_gens.append(float(output.split(" ")[4]))

            # Restore initial stdout
            sys.stdout = stdout_

            # Calculate elapsed time
            times.append(float(time.process_time() - start_time))

            # Store sets of curves
            setsofcurves.append(chlr.set_of_curves)

        # Store sets of curves
        with open("/tmp/artifacts/setsofcurves.pkl", "wb") as f:
            pickle.dump(setsofcurves, f, pickle.HIGHEST_PROTOCOL)

        # Store benchmark results
        f = open("/tmp/artifacts/benchmark_results.md", "w+")
        f.write("# Iterations\n")
        f.write(" * Number: {}\n".format(iterations))
        f.write("# Number of Generations\n")
        f.write(" * Minimum: {}\n".format(min(nb_gens)))
        f.write(" * Maximum: {}\n".format(max(nb_gens)))
        f.write(" * Mean: {}\n".format(round(sum(nb_gens) / len(nb_gens), 2)))
        f.write(" * Std. Dev.: {}\n".format(round(np.std(nb_gens), 2)))
        f.write("# Computing Time\n")
        f.write(" * Minimum: {}\n".format(round(min(times), 2)))
        f.write(" * Maximum: {}\n".format(round(max(times), 2)))
        f.write(" * Mean: {}\n".format(round(sum(times) / len(times), 2)))
        f.write(" * Std. Dev.: {}\n".format(round(np.std(times), 2)))
