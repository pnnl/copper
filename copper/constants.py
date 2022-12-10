"""
constants.py
====================================
Holds all the constants referenced in copper.
TODO: there are additional constants, but i haven't found them yet.
"""

KBTU_TO_KW = 3.412141633
TON_TO_KBTU = 12.0
KILO = 1000
kEER = TON_TO_KBTU * KILO
FIVE_OVER_NINE = 5 / 9
NINE_OVER_FIVE = FIVE_OVER_NINE**-1
TEMP_CONSTANT = 32

# Constants come from the ahri 550/590 Table 1.
# https://www.chiltrix.com/documents/AHRI_Standard_550-590_I-P_2018.pdf
# Abbreviations
# ---------------------
# ECT -> Entering Condenser Temperature (F)
# LCT -> Leaving Condenser Temperature (F)
TOWER_ECT_550_590 = 85.00
TOWER_LCT_550_590 = 94.30

COOLING_ECT_550_590 = 54.00
COOLING_LCT_550_590 = 44.00

EVAP_ECT = 95.00  # TODO: not sure about this one.

# Constants from the ahri 551/591 Table 4.
# https://www.ahrinet.org/sites/default/files/2022-06/AHRI%20Standard%20551-591%202020%20%28SI%29%20with%20Addendum%201.pdf
TOWER_ECT_551_591 = 35.00
TOWER_ENTERING_TEMP_551_591 = 30.00
COOLING_LCT_551_591 = 7.0

# Format for the logger when it logs messages
LOGGING_FORMAT = "%(filename)s:%(lineno)d:%(levelname)s:%(message)s"
