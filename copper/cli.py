"""
cli.py
====================================
This the command line interface module of Copper. It faciliate the integration of Copper into different workflow by being about to run some of Copper's functionality through command line.
"""

import click, json, inspect

from copper.chiller import Chiller
import copper.schema


@click.group()
def cli():
    """
    Copper

    A performance curve generator for building energy simulation
    """


@cli.command()
@click.argument("input_file", type=click.File("rb"), required=True)
def run(input_file):
    """Run a set of Copper instructions through a JSON input file. See 'Using Copper's command line interface in the Quickstart Guide section of the documenation for more information."""

    try:
        f = json.load(input_file)
    except:
        raise ValueError("Could not read the input file. A JSON file is expected.")

    # Validate input file
    if copper.Schema(f).validate():
        for action in f["actions"]:
            eqp_props = action["equipment"]
            # Make sure that the equipment is supported by Copper
            assert eqp_props["type"].lower() in [
                "chiller"
            ], "Equipment type not currently supported by Copper."

            # Get properties for equipment type
            eqp_type_props = inspect.getfullargspec(eval(eqp_props["type"]).__init__)[0]

            # Set the equipment properties from input file
            obj_args = {}
            for p in eqp_type_props:
                if p in list(eqp_props.keys()):
                    obj_args[p] = eqp_props[p]

            # Create instance of the equipment
            obj = eval(eqp_props["type"])(**obj_args)

            # Perform actions defined in input file
            func = action["function_call"]["function"]
            del action["function_call"]["function"]
            args = action["function_call"]
            getattr(obj, func)(**args)


if __name__ == "__main__":
    cli()
