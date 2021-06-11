import unittest
from unittest import TestCase
import copper as cp
import numpy as np

class TestAggregate(TestCase):
    def test_agg(self):
        filters = [("eqp_type", "chiller"),
                   ("sim_engine", "energyplus"),
                   ("model", "ect_lwt"),
                   ("cond_type", "water")]

        #Run unittest with a centrifugal chiller
        lib = cp.Library(path="./fixtures/chiller_curves.json")
        ep_wtr_screw = lib.find_set_of_curvess_from_lib(
            filters=filters + [("source", "EnergyPlus chiller dataset"), ("comp_type", "screw")])
        ep_wtr_scroll = lib.find_set_of_curvess_from_lib(
            filters=filters + [("source", "EnergyPlus chiller dataset"), ("comp_type", "scroll")])
        centrifugal_chlr = ep_wtr_screw + ep_wtr_scroll

        sets = centrifugal_chlr
        water_cooled_curves = cp.SetsofCurves(sets=sets, eqp_type='chiller')
        # remove sets
        for cset in sets:
            cset.remove_curve('eir-f-plr-dt')

        ranges = {
            'eir-f-t': {
                'vars_range': [(4, 10), (10.0, 40.0)],
                'normalization': (6.67, 29.44)
            },
            'cap-f-t': {
                'vars_range': [(4, 10), (10.0, 40.0)],
                'normalization': (6.67, 29.44)
            },
            'eir-f-plr': {
                'vars_range': [(0.0, 1.0)],
                'normalization': (1.0)
            }
        }

        misc_attr = {
            'model': "ect_lwt",
            'ref_cap': 200,
            'ref_cap_unit': "",
            'ref_eff': 5.0,
            'ref_eff_unit': "",
            'comp_type': "aggregated",
            'cond_type': "water",
            'comp_speed': "aggregated",
            'sim_engine': "energyplus",
            'min_plr': 0.1,
            'min_unloading': 0.1,
            'max_plr': 1,
            'name': "Aggregated set of curves",
            'validity': 3,
            'source': "Aggregation"
        }

        #checking with an empty target, we expect a Value Error
        with self.assertRaises(ValueError):
            _, _ = water_cooled_curves.nearest_neighbor_sort()

        #checking with bad target variables that are not present in library
        with self.assertRaises(AssertionError):
            _, _ = water_cooled_curves.nearest_neighbor_sort(target_attr=misc_attr,
                                                             vars=["bad_targets", "wrong_targets"])

        #first look at the test with weighted average
        df, best_idx = water_cooled_curves.nearest_neighbor_sort(target_attr=misc_attr)
        self.assertEqual(best_idx, 8) #the best index for this test is 8
        self.assertEqual(np.round(df.loc[best_idx, 'score'], 3), 0.068)

        #look at the nearest neighbor-implementation with N nearest neighbor, N=7
        df, best_idx = water_cooled_curves.nearest_neighbor_sort(target_attr=misc_attr, N=7)
        score = df.loc[[best_idx], ['score']]['score'].values[0]
        self.assertEqual(best_idx, 8)  # the best index for this test is STILL 8
        self.assertEqual(np.round(score, 3), 0.159)
