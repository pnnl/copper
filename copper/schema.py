"""
schema.py
====================================
Validation of the CLI input files.
"""

import json, jsonschema, logging
from copper.constants import SCHEMA_PATH
from copper.constants import CHILLER_SCHEMA_PATH
from copper.constants import CHILLER_GENE_SCHEMA_PATH
from copper.constants import CHILLER_ACTION_SCHEMA_PATH

# Load schemas
schema_chiller = json.load(open(CHILLER_SCHEMA_PATH, "r"))
schema_chiller_gene = json.load(open(CHILLER_GENE_SCHEMA_PATH, "r"))
schema_chiller_action = json.load(open(CHILLER_ACTION_SCHEMA_PATH, "r"))
schema = json.load(open(SCHEMA_PATH, "r"))

# Define schema store for the validator
schema_store = {
    "copper.chiller.schema.json": schema_chiller,
    "copper.chiller.generate_set_of_curves.schema.json": schema_chiller_gene,
    "copper.chiller.action.schema.json": schema_chiller_action,
    "copper.schema.json": schema,
}


class Schema:
    def __init__(self, input):
        self.input = input
        self.schema = schema
        self.schema_store = schema_store
        self.resolver = jsonschema.RefResolver.from_schema(schema, store=schema_store)
        Validator = jsonschema.validators.validator_for(schema)
        self.validator = Validator(self.schema, resolver=self.resolver)

    def validate(self):
        """Validate input file to be used in the CLI.

        :return: Result of the validation
        :rtype: bool

        """
        try:
            self.validator.validate(self.input)
            return True
        except jsonschema.ValidationError:
            logging.critical(
                "Validation of the input file failed. Please review the input file schema."
            )
            return False
