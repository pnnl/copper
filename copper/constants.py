"""
constants.py
====================================
Holds all the constants referenced in copper.
"""

import os

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
CHILLER_SCHEMA_PATH = os.path.join("./schema/", "copper.chiller.schema.json")
CHILLER_GENE_SCHEMA_PATH = os.path.join(
    "./schema/", "copper.chiller.generate_set_of_curves.schema.json"
)
CHILLER_ACTION_SCHEMA_PATH = os.path.join(
    "./schema/", "copper.chiller.action.schema.json"
)
SCHEMA_PATH = os.path.join("./schema/", "copper.schema.json")
