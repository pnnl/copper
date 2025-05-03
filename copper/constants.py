"""
constants.py
====================================
Holds all the constants referenced in copper.
"""

import os

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
ROOT_DIR = os.path.dirname(__file__)
CHILLER_SCHEMA_PATH = f"{ROOT_DIR}/data/schema/copper.chiller.schema.json"
UNITARYDIRECTEXPANSION_SCHEMA_PATH = (
    f"{ROOT_DIR}/data/schema/copper.unitarydirectexpansion.schema.json"
)
CHILLER_GENE_SCHEMA_PATH = (
    f"{ROOT_DIR}/data/schema/copper.chiller.generate_set_of_curves.schema.json"
)
UNITARYDIRECTEXPANSION_GENE_SCHEMA_PATH = f"{ROOT_DIR}/data/schema/copper.unitarydirectexpansion.generate_set_of_curves.schema.json"
CHILLER_ACTION_SCHEMA_PATH = f"{ROOT_DIR}/data/schema/copper.chiller.action.schema.json"
UNITARYDIRECTEXPANSION_ACTION_SCHEMA_PATH = (
    f"{ROOT_DIR}/data/schema/copper.unitarydirectexpansion.action.schema.json"
)
SCHEMA_PATH = f"{ROOT_DIR}/data/schema/copper.schema.json"
