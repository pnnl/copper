import json, jsonschema, os
import copper as cp
from unittest import TestCase


class TestCurves(TestCase):
    def test_good_schema(self):
        input_file = json.load(open("./tests/data/cli_input_file.json", "r"))
        assert cp.Schema(input=input_file).validate()

    def test_bad_schema(self):
        input_file = json.load(open("./tests/data/cli_input_file.json", "r"))
        input_file["actions"][0]["function_call"]["vars"] = 42.0
        with self.assertLogs() as captured:
            assert cp.Schema(input=input_file).validate() == False
            self.assertTrue(
                captured[0][0].msg
                == "Validation of the input file failed. Please review the input file schema."
            )
