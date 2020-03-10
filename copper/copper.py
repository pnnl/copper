import json


class Chiller:
    def __init__(
        self,
        ref_cap,
        ref_cap_unit,
        full_eff,
        full_eff_unit,
        part_eff,
        part_eff_unit,
        compressor_type,
        condenser_type,
        compressor_speed,
        curveset="",
        model="ect&lwt",
        sim_engine="energyplus",
    ):
        self.compressor_type = compressor_type
        self.condenser_type = condenser_type
        self.compressor_speed = compressor_speed
        self.ref_cap = ref_cap
        self.ref_cap_unit = ref_cap_unit
        self.full_eff = full_eff
        self.full_eff_unit = full_eff_unit
        self.part_eff = part_eff
        self.part_eff_unit = part_eff_unit
        self.model = model
        self.sim_engine = sim_engine

    def generate_curve_set(self):
        """
        Generate a curve set that matches a targeted full and part load efficiency.
        The curves are generated to be used with the specified energy modeling software and with the specified algorithm.
        """
        pass

    def find_base_curves(self):
        """
        Find an existing equipment curve that best matches the chiller
        """
        # Define instance of the GA to use lookup function
        ga = GA()

        # Define chiller properties
        props = [
            ("eqp_type", "chiller"),
            ("compressor_type",self.compressor_type),
            ("condenser_type", self.condenser_type),
            ("compressor_speed", self.compressor_speed),
            ("sim_engine", self.sim_engine),
            ("algorithm", self.model)]

        # Find equipment match in the library
        eqp_match = ga.find_equipment(filters=props)

        if len(eqp_match) > 0:
            # If multiple equipment match the specified properties,
            # return the one that has numeric attributes that best
            # match the proposed case
            if len(eqp_match) > 1:
                return ga.get_curve_set_by_name(ga.get_best_match(self, eqp_match))
            else:
                return ga.get_curve_set_by_name(eqp_match)
        else:
            raise ValueError('Could not find a set of curves that matches the specified properties.')


class CurveSet:
    def __init__(self):
        self.name = ""
        self.curves = []

    def generate(self, sim_engine, algorithm):
        pass

    def plot(self, unit):
        pass


class Curve:
    def __init__(self):
        self.output_var = ""
        self.curve_type = ""
        self.x_min = 0
        self.y_min = 0
        self.x_max = 0
        self.y_max = 0
        self.out_min = 0
        self.out_max = 0
        self.coeffs = []

    def evaluate(self, x, y):
        """
        Return the output of a curve.
        """
        x = min(max(x, self.x_min), self.x_max)
        y = min(max(y, self.y_min), self.y_max)

        if self.curve_type == "bi_quad":
            out = (
                self.coeffs[0]
                + self.coeffs[1] * x
                + self.coeffs[2] * x ** 2
                + self.coeffs[3] * y
                + self.coeffs[4] * y ** 2
                + self.coeffs[5] * x * y
            )
            return min(max(out, self.out_min), self.out_max)
        if self.curve_type == "bi_cub":
            out = (
                self.coeffs[0]
                + self.coeffs[1] * x
                + self.coeffs[2] * x ** 2
                + self.coeffs[3] * y
                + self.coeffs[4] * y ** 2
                + self.coeffs[5] * x * y
                + self.coeffs[6] * x ** 3
                + self.coeffs[7] * y ** 3
                + self.coeffs[8] * y * x ** 2
                + self.coeffs[9] * x * y ** 2
            )
            return min(max(out, self.out_min), self.out_max)
        if self.curve_type == "quad":
            out = (
                self.coeffs[0]
                + self.coeffs[1] * x
                + self.coeffs[2] * x ** 2
            )
            return min(max(out, self.out_min), self.out_max)


class Unit:
    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

    def conversion(self, new_unit):
        """
        Convert efficiency rating
        """
        if new_unit == "kWpton":
            if self.unit == "COP":
                return 12.0 / (self.value * 3.412)
            elif self.unit == "kWpton":
                return self.value
            elif self.unit == "EER":
                return 12.0 / self.value
            else:
                return self.value
        elif new_unit == "COP":
            if self.unit == "kWpton":
                return 12.0 / self.value / 3.412
            elif self.unit == "COP":
                return self.value
            elif self.unit == "EER":
                return self.value / 3.412
            else:
                return self.value
        elif new_unit == "EER":
            if self.unit == "kWpton":
                return 12.0 / self.value
            elif self.unit == "EER":
                return self.value
            elif self.unit == "COP":
                return 3.412 * self.value
            else:
                return self.value
        elif new_unit == "ton":
            if self.unit == "kW":
                return self.value * (3412 / 12000)
        elif new_unit == "kW":
            if self.unit == "ton":
                return self.value / (3412 / 12000)

class GA:
    def __init__(self,):
        pass

    def iplv(self, equip, unit="kWpton"):
        """
        Calculate equipment IPLV
        """

        # Retrieve equipment efficiency and unit
        kwpton_ref = equip.full_eff
        kwpton_ref_unit = equip.full_eff_unit

        # Convert to kWpton if necessary
        if equip.full_eff_unit != "kWpton":
            kwpton_ref_unit = Unit(kwpton_ref, kwpton_ref_unit)
            kwpton_ref = kwpton_ref_unit.conversion("kWpton")
        
        # Full load conditions
        load_ref = 1
        eir_ref = 1 / (12 / kwpton_ref / 3.412141633)    

        # Test conditions
        # Same approach as EnergyPlus
        # Same as AHRI Std 550/590
        loads = [1, 0.75, 0.5, 0.25]

        # List of equipment efficiency for each load
        kwpton_lst = []

        # DOE-2 chiller model
        if equip.model == "ect&lwt":
            if equip.condenser_type == 'air':
                # Temperatures from AHRI Std 550/590
                chw = 6.67
                ect = [3 + 32 * loads[0],
                       3 + 32 * loads[1],
                       3 + 32 * loads[2],
                       13]
            elif equip.condenser_type == 'water':
                # Temperatures from AHRI Std 550/590
                chw = 6.67
                ect = [8 + 22 * loads[0],
                       8 + 22 * loads[1],
                       19, 19]
            
            # Retrieve curves
            for curve in equip.curveset.curves:
                if curve.output_var == "cap-f-T":
                    cap_f_t = curve
                elif curve.output_var == "eir-f-T":
                    eir_f_t = curve
                else:
                    eir_f_plr = curve

            # Calculate EIR for each testing conditions
            for idx, load in enumerate(loads):
                dt = ect[idx] - chw
                cap_f_chw_ect = cap_f_t.evaluate(chw, ect[idx])
                eir_f_chw_ect = eir_f_t.evaluate(chw, ect[idx])
                cap_op = load_ref * cap_f_chw_ect
                plr = load / cap_op
                eir_plr = eir_f_plr.evaluate(plr,dt)
                eir = eir_ref * eir_f_chw_ect * eir_plr / plr
                kwpton_lst.append(eir) # / 3.412141633 * 12)

            # Coefficients from AHRI Std 550/590
            iplv = 1 / ((0.01 / kwpton_lst[0]) + (0.42 / kwpton_lst[1]) + (0.45 / kwpton_lst[2]) + (0.12 / kwpton_lst[3]))

        else:
            # TODO:
            # Implement IPLV calcs for other chiller algorithm
            raise ValueError("Algorithm not implemented.")

        # Convet IPLV to desired unit
        if unit != "kWpton":
            iplv_org = Unit(iplv, "kWpton")
            iplv = iplv_org.conversion(unit)

        return iplv


    def kwpton(self, value):
        pass

    def find_equipment(
        self,
        db_path="./fixtures/curves_from_csv.json",
        filters=[],
    ):
        """
        Find equipment matching specified filter in the curve library
        """
        # Load curve database
        c_db = json.loads(open(db_path, "r").read())

        eqp_match_dict = {}
        for eqp in c_db:
            # Check if equipment properties match specified filter
            if all(c_db[eqp][prop] == val for prop, val in filters):
                eqp_match_dict[eqp] = c_db[eqp]

        return eqp_match_dict

    def get_best_match(self, eqp, matches):
        # Initialize numeric attribute difference
        diff = float("inf")

        # Iterate over matches and calculate the 
        # difference in numeric fields
        for name, val in matches.items():
            # Retrive full load/reference numeric attribute
            cap = matches[name]['ref_cap']
            cap_unit = matches[name]['ref_cap_unit']
            eff = matches[name]['full_eff']
            eff_unit = matches[name]['full_eff_unit']

            # Capacity conversion
            if cap_unit != eqp.ref_cap_unit:
                c_unit = Unit(cap, cap_unit)
                cap = c_unit.conversion(eqp.ref_cap_unit)
                cap_unit = eqp.ref_cap_unit

            # Efficiency conversion
            if eff_unit != eqp.full_eff_unit:
                c_unit = Unit(eff, eff_unit)
                eff = c_unit.conversion(eqp.full_eff_unit)
                eff_unit = eqp.full_eff_unit

            # Compute difference
            c_diff = abs((eqp.ref_cap - cap) / eqp.ref_cap) + \
                abs((eqp.full_eff - eff) / eqp.full_eff)

            if c_diff < diff:
                # Update lowest numeric difference
                diff = c_diff

                # Update best match
                best_match = name
        
        # TODO;
        # Remove following line, just for debugging
        print(best_match)
        
        return best_match

    def get_curve_set_by_name(self, name, db_path="./fixtures/curves_from_csv.json"):
        """
        Retrieve curve set from the library by name
        """
        # Load curve database
        c_db = json.loads(open(db_path, "r").read())

        # Initialize curve set object
        c_set = CurveSet()
        c_set.name = name
        
        # List of curves
        c_lst = []

        # Define curve objects
        for c in c_db[name]["curves"]:
            c_lst.append(self.get_curve(c_db, c, c_db[name]))

        # Add curves to curve set object
        c_set.curves = c_lst

        return c_set

    def get_curve(self, c_db, c, c_name):
        # Initialize curve object
        c_obj = Curve()
        c_obj.output_var = c

        # Curve properties
        c_prop = c_name["curves"][c]

        # Retrive all attributes of the curve object
        for c_att in list(Curve().__dict__):
            # Set the attribute of new Curve object 
            # if attrubute are identified in database entry
            if c_att in list(c_prop.keys()):
                c_obj.__dict__[c_att] = c_prop[c_att]

        return c_obj

    def get_curve_sets(
        self,
        db_path="./fixtures/curves_from_csv.json",
        filters=[],
    ):
        """
        Retrive curve sets fromt he libraray that match the specified filters (tuples of strings)
        """
        # Load curve database
        c_db = json.loads(open(db_path, "r").read())
        
        # Find name of equiment that match specified filter
        eqp_match = self.find_equipment(db_path, filters)

        # List of curve sets that match specified filters
        curve_sets = []

        # Retrieve identified equipment's curve sets from the library
        for name, props in eqp_match.items():
            c_set = CurveSet()
            c_set.name = name
            c_lst = []
            # Create new CurveSet and Curve objects for all the 
            # sets of curves identified as matching the filters
            for c in c_db[name]["curves"]:
                c_lst.append(self.get_curve(c_db, c, c_db[name]))
            c_set.curves = c_lst
            curve_sets.append(c_set)

        return curve_sets

    def generate_population():
        pass

    def evolve_population():
        pass

    def determine_fitness():
        pass
    
    def grade_population():
        pass

    def random_selection():
        pass

    def perform_mutation():
        pass

    def perform_crossover():
        pass

    def identify_best_performer():
        pass



