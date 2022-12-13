from unittest import TestCase
import copper as cp
import os

LOCATION = os.path.dirname(os.path.realpath(__file__))
CHILLER_LIB = os.path.join(LOCATION, "../copper/lib", "chiller_curves.json")


class TestCLI(TestCase):
    def setUp(self) -> None:
        """Runs before every test. Good place to initialize values and store common objects."""
        self.tolerance = 3

    def tearDown(self) -> None:
        """Runs after every test and cleans up file created from the tests."""
        pass

    def test_no_argument_error(self):
        os.system(f"python {LOCATION}/../copper/cli.py run {CHILLER_LIB}")
