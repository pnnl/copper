from unittest import TestCase
from copper.calculations import *

class TestIPLVCalcs(TestCase):
    def test_iplv_calcs_1(self):
        self.assertTrue(iplv_calcs(0.3, [], 'air_cooled') == 0)