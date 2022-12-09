from unittest import TestCase
import copper as cp

#TODO: look into adding hypothesis to the code to do range checking.

class TestRating(TestCase):

    def setUp(self) -> None:
        """Runs before every test. Good place to initialize values and store common objects.
        """
        self.tolerance = 3

    def tearDown(self) -> None:
        """Runs after every test and cleans up file created from the tests.
        """
        pass

    def test_test_cop_to_kw_per_ton(self):
        self.assertAlmostEqual(cp.newUnits.cop_to_kw_per_ton(14.0), 0.251, places=self.tolerance)

    def test_test_kw_per_ton_to_cop(self):
        self.assertAlmostEqual(cp.newUnits.kw_per_ton_to_cop(14.0), 0.251, places=self.tolerance)
    
    def test_cop_to_eer(self):
        self.assertAlmostEqual(cp.newUnits.cop_to_eer(14.0), 47.76998, places=self.tolerance)

    def test_kw_per_ton_to_eer(self):
        self.assertAlmostEqual(cp.newUnits.kw_per_ton_to_eer(14.0), 0.857, places=self.tolerance)

    def test_eer_to_cop(self):
        self.assertAlmostEqual(cp.newUnits.eer_to_cop(14.0), 4.103, places=self.tolerance)

    def test_eer_to_kw_per_ton(self):
        self.assertAlmostEqual(cp.newUnits.eer_to_kw_per_ton(14.0), 0.857, places=self.tolerance)
    
    def test_kw_to_ton(self):
        self.assertAlmostEqual(cp.newUnits.kw_to_ton(14.0), 3.981, places=self.tolerance)

    def test_watt_to_ton(self):
        self.assertAlmostEqual(cp.newUnits.watt_to_ton(14.0), 0.003981, places=self.tolerance)

    def test_ton_to_kw(self):
        self.assertAlmostEqual(cp.newUnits.ton_to_kw(14.0), 49.2359, places=self.tolerance)

    def test_watt_to_kw(self):
        self.assertAlmostEqual(cp.newUnits.watt_to_kw(14.0), 0.014, places=self.tolerance)
    
    def test_ton_to_watt(self):
        self.assertAlmostEqual(cp.newUnits.ton_to_watt(14.0), 49235.93979, places=self.tolerance)

    def test_F_to_C(self):
        self.assertAlmostEqual(cp.newUnits.F_to_C(14.0), -10.0, places=self.tolerance)

    def test_C_to_F(self):
        self.assertAlmostEqual(cp.newUnits.C_to_F(14.0), 57.199999, places=self.tolerance)


    
