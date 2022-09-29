Chillers
=========

**Copper** can generate performance curves for all types of vapor-compression chillers. It handles two of the chiller models that are implemented in most common building energy simulation software, both models use performance curves that are function of the chilled water leaving temperature and the part load ratio (ratio of load on the chiller to the operating chiller capacity). The main difference between the two models is that one uses performance curves that are function of the entering condenser temperature (refer to thereafter as the ECT model) and the other one is function of the leaving condenser temperature (refer to therafter as the LCT model). Additional documentation for the former can be found in the `EnergyPlus`_ and `DOE-2`_ engineering manuals, documentation for the later can be found in the `EnergyPlus engineering manual`_.

Chiller Data Library
---------------------
The `chiller library`_ contains performance curves for exisiting chillers, both for the ECT and LCT model. The number of performance curves for each compressor/condenser/model type varies, a summary of the library is shown in Table 1.

.. csv-table:: Table 1 - Summary of the Chiller Library
   :file: chiller_lib_summary.csv
   :widths: 20, 20, 20, 20, 20
   :header-rows: 1

.. _EnergyPlus: https://bigladdersoftware.com/epx/docs/8-7/engineering-reference/chillers.html#electric-chiller-model-based-on-condenser-entering-temperature
.. _DOE-2: https://doe2.com/Download/DOE-21E/DOE-2EngineersManualVersion2.1A.pdf
.. _EnergyPlus engineering manual: https://bigladdersoftware.com/epx/docs/8-7/engineering-reference/chillers.html#electric-chiller-model-based-on-condenser-leaving-temperature
.. _chiller library: https://github.com/pnnl/copper/blob/develop/copper/lib/chiller_curves.json