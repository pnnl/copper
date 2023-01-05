from copper.constants import LOGGING_FORMAT, SCHEMA_LOCATION
import jsonschema
from jsonschema import validate
import json
import logging

logging.basicConfig(format=LOGGING_FORMAT)


class InstructionSet:
    def __init__(self, instruction_set_file) -> None:
        self.instruction_set_file = instruction_set_file
        self.error_message = ""
        self.instruction_set_content = self._read_instruction_set()
        if not self._is_instruction_set_valid():
            raise jsonschema.exceptions.ValidationError(self.error_message)

    def _read_instruction_set(self):
        with open(self.instruction_set_file, "r") as f:
            instruction_set_data = json.load(f)
        return instruction_set_data

    def _is_instruction_set_valid(self):
        with open(SCHEMA_LOCATION, "r") as f:
            schema_data = json.load(f)
        try:
            validate(instance=self.instruction_set_content, schema=schema_data)
        except jsonschema.exceptions.ValidationError as err:
            self.error_message = err.message
            logging.critical(f"{self.error_message}")
            return False
        return True


if __name__ == "__main__":
    instruction_set = InstructionSet("tests/data/input_example.json")
