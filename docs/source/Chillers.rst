Chillers
=========

**Copper** can generate performance curves for all types of vapor-compression chillers. It handles two of the chiller models that are implemented in most common building energy simulation software: a entering condenser temperature-based model (ECT model, implemented in EnergyPlus, TRNSYS, and DOE-2) and a leaving condenser temperature-based model (LCT model, implemented in EnergyPlus).

Chiller Data Library
---------------------
The `chiller library`_ contains performance curves for exisiting chillers, both for the ECT and LCT model. The number of performance curves for each compressor/condenser/model type varies, a summary of the library is shown in Table 1.

.. csv-table:: Table 1 - Summary of the Chiller Library
   :file: chiller_lib_summary.csv
   :widths: 20, 20, 20, 20, 20
   :header-rows: 1

.. _chiller library: https://github.com/pnnl/copper/blob/develop/copper/lib/chiller_curves.json