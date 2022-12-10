Input Guide
===========
The command line argument tool reads in a json file of the following format...

.. code-block:: json

    {
        "Curve Name":{
            "eqp_type":             "string",
            "compressor_type":      "string",
            "condenser_type":       "string",
            "compressor_speed":     "string",
            "ref_cap":              "float",
            "ref_cap_unit":         "string",
            "full_eff":             "float",
            "full_eff_unit":        "string",
            "part_eff":             "float",
            "part_eff_unit":        "string",
            "sim_engine":           "string",
            "model":                "string",
            "do":{
                "generate_set_of_curves":{
                    "vars": [
                                "string"
                            ],
                    "method":           "string",
                    "tol":              "float",
                    "export_path":      "float",
                    "export_format":    "float",
                    "export_name":      "float",
                    "random_seed":      "integer"
                }
            }
        }
    }


.. list-table:: Table of Acceptable Values
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter Name
     - Parameter Values
     - Description

   * - eqp_type
     - chiller
     - Type of equipment being analyzed.

   * - compressor_type
     - any |br| centrifugal |br| screw |br| scroll
     - Type of compressor being analyzed

   * - condenser_type
     - air |br| water
     - Either an air-cooled or water-cooled condenser

   * - compressor_speed
     - any |br| constant |br| variable
     - Rate the compressor moves at

   * - ref_cap
     - > 0
     - TODO: i don't know

   * - ref_cap_unit
     - eer |br| kW |br| kw/ton |br| ton |br| null
     - TODO: i don't know

   * - full_eff
     - > 0
     - TODO: i don't know

   * - full_eff_unit
     - eer |br| kw/ton |br| cop |br| null
     - TODO: i don't know

   * - part_eff
     - > 0
     - TODO: i don't know

   * - part_eff_unit
     - eer |br| kw/ton |br| cop
     - TODO: i don't know

   * - sim_engine
     - energyplus
     - TODO: i don't know

   * - model
     - ect_lwt |br| lct_lwt
     - TODO: i don't know

   * - do
     - {}
     - TODO: i don't know

   * - generate_set_of_curves
     - {}
     - TODO: i don't know

   * - vars
     - eir-f-t |br| cap-f-t |br| eir-f-plr
     - TODO: i don't know

   * - method
     - best_match |br| nearest_neighbor |br| weighted_average
     - TODO: i don't know

   * - tol
     - > 0
     - TODO: i don't know

   * - export path
     - Any valid string
     - TODO: i don't know

   * - export_format
     - csv |br| idf |br| json
     - TODO: i don't know

   * - export_name
     - Any valid string
     - TODO: i don't know

   * - random_seed
     - > 0
     - TODO: i don't know




.. list-table:: Table of Output
   :widths: 25 25
   :header-rows: 1

   * - out_var
     - eir-f-t |br| cap-f-t |br| eir-f-plr

   * - type
     - bi_quad

   * - ref_evap_fluid_flow
     - >= 0 |br| null

   * - ref_cond_fluid_flow
     - >= 0 |br| null

   * - ref_lwt
     - >= 0 |br| null

   * - ref_ect
     - >= 0 |br| null

   * - ref_lct
     - >= 0 |br| null

   * - units
     - si

   * - x_min
     - >= 0 |br| null

   * - y_min
     - >= 0 |br| null

   * - x_max
     - > x_min |br| null

   * - y_max
     - > y_min |br| null

   * - out_min
     - >= 0 |br| null

   * - out_max
     - >= out_min |br| null

   * - coeff1
     - >= 0 |br| null

   * - coeff2
     - >= 0 |br| null

   * - coeff3
     - >= 0 |br| null

   * - coeff4
     - >= 0 |br| null

   * - coeff5
     - >= 0 |br| null

   * - coeff6
     - >= 0 |br| null

   * - coeff7
     - >= 0 |br| null

   * - coeff8
     - >= 0 |br| null

   * - coeff9
     - >= 0 |br| null

   * - coeff10
     - >= 0 |br| null


.. # define a hard line break for HTML
.. |br| raw:: html

   <br />