"""
constants.py
====================================
Holds all the constants referenced in copper.
"""

import os
from pathlib import Path

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
filep = Path(__file__)
CHILLER_SCHEMA_PATH = os.path.join(filep, "/schema/", "copper.chiller.schema.json")
CHILLER_GENE_SCHEMA_PATH = os.path.join(
    filep, "/schema/", "copper.chiller.generate_set_of_curves.schema.json"
)
CHILLER_ACTION_SCHEMA_PATH = os.path.join(
    filep, "/schema/", "copper.chiller.action.schema.json"
)
SCHEMA_PATH = os.path.join(filep, "/schema/", "copper.schema.json")
