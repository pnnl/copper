Chillers
=========

**Copper** can generate performance curves for all types of vapor-compression chillers. It handles two of the chiller models that are implemented in most common building energy simulation software tools; both models use performance curves that are function of the temperature of the water leaving the chiller and the part load ratio (ratio of load on the chiller to the operating chiller capacity). The main difference between the two models is that one uses performance curves that are function of the entering condenser temperature (referred to herein as the ECT model) and the other is function of the leaving condenser temperature (referred to herein as the LCT model). Additional documentation on the ECT model can be found in the `EnergyPlus`_ and `DOE-2`_ engineering manuals; documentation on the LCT model can be found in the `EnergyPlus engineering manual`_.

Chiller data library
---------------------
The `chiller library`_ contains performance curves for existing chillers for both the ECT model and the LCT model. The number of performance curves for each compressor/condenser/model type varies; Table 1 provides a summary of the library.

.. csv-table:: Table 1 - Summary of the Chiller Library
   :file: chiller_lib_summary.csv
   :widths: 20, 20, 20, 20, 20
   :header-rows: 1

.. _EnergyPlus: https://bigladdersoftware.com/epx/docs/8-7/engineering-reference/chillers.html#electric-chiller-model-based-on-condenser-entering-temperature
.. _DOE-2: https://doe2.com/Download/DOE-21E/DOE-2EngineersManualVersion2.1A.pdf
.. _EnergyPlus engineering manual: https://bigladdersoftware.com/epx/docs/8-7/engineering-reference/chillers.html#electric-chiller-model-based-on-condenser-leaving-temperature
.. _chiller library: https://github.com/pnnl/copper/blob/develop/copper/lib/chiller_curves.json