from unittest import TestCase
from copper.read_instruction_set import InstructionSet
import jsonschema

#TODO: look into adding hypothesis to the code to do range checking.

class TestInstructionSet(TestCase):

    def setUp(self) -> None:
        """Runs before every test. Good place to initialize values and store common objects.
        """
        self.happy_path_json_input = "tests/data/input_example.json"
        self.unhappy_path_json_input = "tests/data/input_example_missing_required.json"

    def tearDown(self) -> None:
        """Runs after every test and cleans up file created from the tests.
        """
        pass
    
    def test_read_valid_instruction_set(self):
        """Checks that a valid instruction set can be read in.
        """
        self.assertTrue(InstructionSet(self.happy_path_json_input))

    def test_read_instruction_set_missing_require_parameter(self):
        """Checks an instruction that is missing a required value will raise an error.
        """
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            InstructionSet(self.unhappy_path_json_input)