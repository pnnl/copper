"""
constants.py
====================================
Holds all the constants referenced in copper.
"""

import os

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
ROOT_DIR = os.path.dirname(__file__)
CHILLER_SCHEMA_PATH = f"{ROOT_DIR}/../schema/copper.chiller.schema.json"
CHILLER_GENE_SCHEMA_PATH = (
    f"{ROOT_DIR}/../schema/copper.chiller.generate_set_of_curves.schema.json"
)
CHILLER_ACTION_SCHEMA_PATH = f"{ROOT_DIR}/../schema/copper.chiller.action.schema.json"
SCHEMA_PATH = f"{ROOT_DIR}/../schema/copper.schema.json"
