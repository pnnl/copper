Input Guide
===========
The command line argument tool reads in a json file. of the following format.

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
   :widths: 25 25
   :header-rows: 1

   * - Parameter Name
     - Parameter Values

   * - eqp_type
     - chiller

   * - compressor_type
     - any, centrifugal, screw, scroll

   * - condenser_type
     - air, water

   * - compressor_speed
     - any, constant, variable

   * - ref_cap
     - > 0

   * - ref_cap_unit
     - eer, kW, kw/ton, ton, null

   * - full_eff
     - > 0

   * - full_eff_unit
     - eer, kw/ton, cop, null

   * - part_eff
     - > 0

   * - part_eff_unit
     - eer, kw/ton, cop

   * - sim_engine
     - energyplus

   * - model
     - ect_lwt, lct_lwt

   * - do
     - TODO: does not exist in sample instruction set.

   * - set_of_curves
     - dictionary with keys of eir-f-t, cap-f-t, eir-f-plr

   * - out_var
     - test

   * - type
     - test

   * - ref_evap_fluid_flow
     - test

   * - ref_cond_fluid_flow
     - test

   * - ref_lwt
     - test

   * - ref_ect
     - test

   * - ref_lct
     - test

   * - units
     - test

   * - x_min
     - test

   * - y_min
     - test

   * - x_max
     - test

   * - y_max
     - test

   * - out_min
     - test

   * - out_max
     - test

   * - coeff1
     - test

   * - coeff2
     - test

   * - coeff3
     - test