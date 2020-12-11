from unittest import TestCase

import copper as cp
import time, sys, io


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
        iterations = 20

        for _ in range(iterations):
            # Initialise time counter
            start_time = time.process_time()

            # Store initial stdout
            stdout_ = sys.stdout
            sys.stdout = io.StringIO()
            
            # Generate a set of performance curves for targeted equipment
            chlr.generate_set_of_curves(
                vars=["eir-f-t", "cap-f-t", "eir-f-plr"],
                method="typical"
            )

            # Extract number of generation needed to reach goal
            output = sys.stdout.getvalue()
            
            # Restore initial stdout
            sys.stdout = stdout_

            print(float(output.split(" ")[4]))
            nb_gens.append(float(output.split(" ")[4]))

            # Calculate elapsed time
            times.append(float(time.process_time() - start_time))
        
        f = open("/tmp/artifacts/benchmark_results.txt", "w")
        f.write("Avg. generations: {}\nAvg. time: {}".format(round(sum(nb_gens)/len(nb_gens),2), round(sum(times)/len(times),2)))

