import json
from copper.units import *
from copper.curves import *


class Library:
    def __init__(
        self, path="./fixtures/chiller_curves.json"
    ):  # ./fixtures/chiller_curves.json"

        self.path = path
        self.data = json.loads(open(self.path, "r").read())

    def content(self):
        return self.data

    def get_unique_eqp_fields(self):
        """Get all unique values for each field of a particular equipment.

        :return: Dictionary showing all unique values for each equipment field.
        :rtype: dict[str, ]

        """
        # Store all value for each field
        uni_field_val = {}
        for _, eqp_f in self.data.items():
            for field, val in eqp_f.items():
                if field != "curves" and field != "name":
                    # Check if field has already been added
                    if field not in uni_field_val.keys():
                        uni_field_val[field] = [val]
                    else:
                        uni_field_val[field].append(val)
        # Retain only unique values
        for field, val in uni_field_val.items():
            uni_field_val[field] = set(val)
        return uni_field_val

    def find_set_of_curvess_from_lib(self, filters=[]):
        """Retrieve sets of curves from a library matching specific filters.

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :return: All set of curves object matching the filters
        :rtype: list()

        """
        # Find name of equiment that match specified filter
        eqp_match = self.find_equipment(filters)

        # List of sets of curves that match specified filters
        set_of_curvess = []

        # Retrieve identified equipment's sets of curves from the library
        for name, props in eqp_match.items():
            c_set = SetofCurves(props["eqp_type"])

            # Retrive all attributes of the sets of curves object
            for c_att in list(c_set.__dict__):
                # Set the attribute of new Curve object
                # if attrubute are identified in database entry
                if c_att in list(self.data[name].keys()):
                    c_set.__dict__[c_att] = self.data[name][c_att]

            c_lst = []

            # Create new SetofCurves and Curve objects for all the
            # sets of curves identified as matching the filters
            for c in self.data[name]["curves"]:
                c_lst.append(
                    self.get_curve(
                        c, self.data[name], eqp_type=self.data[name]["eqp_type"]
                    )
                )
            c_set.curves = c_lst
            set_of_curvess.append(c_set)

        return set_of_curvess

    def find_equipment(self, filters=[]):
        """Find equipment matching specified filter in the curve library.

        Special filter characters:

        - ~! means "all except..."
        - ! means "do not include..."
        - ~ means "include..."

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :return: Dictionary of field for each equipment matching specified filter
        :rtype: dict[str,dict[]]

        """
        eqp_match_dict = {}
        for eqp in self.data:
            assertions = []
            for prop, val in filters:
                # ~! = all but...
                if "~!" in val:
                    assertions.append(
                        val.replace("~!", "").lower().strip()
                        not in self.data[eqp][prop].lower().strip()
                    )
                # ! = does not include
                elif "!" in val:
                    assertions.append(self.data[eqp][prop] != val)
                # ~ = includes
                elif "~" in val:
                    assertions.append(
                        val.replace("~", "").lower().strip()
                        in self.data[eqp][prop].lower().strip()
                    )
                else:
                    assertions.append(self.data[eqp][prop] == val)
            if all(assertions):
                eqp_match_dict[eqp] = self.data[eqp]

        return eqp_match_dict

    def get_set_of_curves_by_name(self, name):
        """Retrieve set of curves from the library by name.

        :param str name: Curve name
        :return: Set of curves object
        :rtype: SetofCurves()

        """
        # Initialize set of curves object
        c_set = SetofCurves("chiller")
        c_set.name = name

        # List of curves
        c_lst = []
        # Define curve objects
        try:
            for c in self.data[name]["curves"]:
                c_lst.append(
                    self.get_curve(
                        c, self.data[name], eqp_type=self.data[name]["eqp_type"]
                    )
                )
            # Add curves to set of curves object
            c_set.curves = c_lst
            return c_set
        except:
            raise ValueError("Cannot find curve in library.")

    def get_curve(self, c, c_name, eqp_type):
        """Retrieve individual attribute of a curve object.

        :param Curve() c: Curve object
        :param str c_name: Name of the curve object
        :param str eqp_type: Type of equipment associated with the curve
        :return: Curve object
        :rtype: Curve()

        """
        # Curve properties
        c_prop = c_name["curves"][c]
        # Initialize curve object
        c_obj = Curve(eqp_type, c_prop["type"])
        c_obj.out_var = c
        # Retrive all attributes of the curve object
        for c_att in list(Curve(eqp_type, c_prop["type"]).__dict__):
            # Set the attribute of new Curve object
            # if attrubute are identified in database entry
            if c_att in list(c_prop.keys()):
                c_obj.__dict__[c_att] = c_prop[c_att]
        return c_obj

    def find_base_curves(self, filters, eqp):
        """Find an existing equipment curve that best matches the equipment.

        :param list(tuple()) filters: Filter represented by tuples (field, val)
        :param eqp: Equipment object(e.g. Chiller())
        :return: Set of curves object
        :rtype: SetofCurves()

        """
        # Find equipment match in the library
        eqp_match = self.find_equipment(filters=filters)

        if len(eqp_match) > 0:
            # If multiple equipment match the specified properties,
            # return the one that has numeric attributes that best
            # match the proposed case
            if len(eqp_match) > 1:
                return self.get_set_of_curves_by_name(
                    self.get_best_match(eqp, eqp_match)
                )
            else:
                return self.get_set_of_curves_by_name(eqp_match)
        else:
            raise ValueError(
                "Could not find a set of curves that matches the specified properties."
            )

    def get_best_match(self, eqp, matches):
        """Find the set of curves matching the equipment characteristics the best.

        :param eqp: Equipment object(e.g. Chiller())
        :param dict[str,dict[]] matches: All potential matches
        :return: Name of the set of curves that best matches the equipment characteristics
        :rtype: str

        """
        # Initialize numeric attribute difference
        diff = float("inf")

        # Iterate over matches and calculate the
        # difference in numeric fields
        for name, val in matches.items():
            # Retrieve full load/reference numeric attribute
            if eqp.type == "chiller":
                cap = val["ref_cap"]
                cap_unit = val["ref_cap_unit"]
                eff = val["ref_eff"]
                eff_unit = matches[name]["ref_eff_unit"]

                if not cap is None:
                    # Capacity conversion
                    if cap_unit != eqp.ref_cap_unit:
                        c_unit = Units(cap, cap_unit)
                        cap = c_unit.conversion(eqp.ref_cap_unit)
                        cap_unit = eqp.ref_cap_unit

                    # Efficiency conversion
                    if eff_unit != eqp.full_eff_unit:
                        c_unit = Units(eff, eff_unit)
                        eff = c_unit.conversion(eqp.full_eff_unit)
                        eff_unit = eqp.full_eff_unit

                        # Compute difference
                        c_diff = abs((eqp.ref_cap - cap) / eqp.ref_cap) + abs(
                            (eqp.full_eff - eff) / eqp.full_eff
                        )

                        if c_diff < diff:
                            # Update lowest numeric difference
                            diff = c_diff

                            # Update best match
                            best_match = name

        return best_match