import json, jsonschema, os
from unittest import TestCase


input_file = json.load(open("./tests/data/cli_input_file.json", "r"))
CHILLER_SCHEMA_PATH = os.path.join("./schema/", "copper.chiller.schema.json")
CHILLER_GENE_SCHEMA_PATH = os.path.join(
    "./schema/", "copper.chiller.generate_set_of_curves.schema.json"
)
CHILLER_ACTION_SCHEMA_PATH = os.path.join(
    "./schema/", "copper.chiller.action.schema.json"
)
SCHEMA_PATH = os.path.join("./schema/", "copper.schema.json")
schema_chiller = json.load(open(CHILLER_SCHEMA_PATH, "r"))
schema_chiller_gene = json.load(open(CHILLER_GENE_SCHEMA_PATH, "r"))
schema_chiller_action = json.load(open(CHILLER_ACTION_SCHEMA_PATH, "r"))
schema = json.load(open(SCHEMA_PATH, "r"))

schema_store = {
    "copper.chiller.schema.json": schema_chiller,
    "copper.chiller.generate_set_of_curves.schema.json": schema_chiller_gene,
    "copper.chiller.action.schema.json": schema_chiller_action,
    "copper.schema.json": schema,
}


class TestCurves(TestCase):
    def test_chiller_schema(self):
        resolver = jsonschema.RefResolver.from_schema(
            schema_chiller, store=schema_store
        )
        Validator = jsonschema.validators.validator_for(schema_chiller)
        validator = Validator(schema_chiller, resolver=resolver)
        validator.validate(input_file["actions"][0]["equipment"])

    def test_chiller_generate_set_of_curves(self):
        resolver = jsonschema.RefResolver.from_schema(
            schema_chiller_gene, store=schema_store
        )
        Validator = jsonschema.validators.validator_for(schema_chiller_gene)
        validator = Validator(schema_chiller_gene, resolver=resolver)
        validator.validate(input_file["actions"][0]["function_call"])

    def test_chiller_action(self):
        resolver = jsonschema.RefResolver.from_schema(
            schema_chiller_action, store=schema_store
        )
        Validator = jsonschema.validators.validator_for(schema_chiller_action)
        validator = Validator(schema_chiller_action, resolver=resolver)
        validator.validate(input_file["actions"][0])

    def test_copper(self):
        resolver = jsonschema.RefResolver.from_schema(schema, store=schema_store)
        Validator = jsonschema.validators.validator_for(schema)
        validator = Validator(schema, resolver=resolver)
        validator.validate(input_file)
