from setuptools import setup, find_packages
from pathlib import Path

# read the contents of the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name="copper-bem",
    version="0.1.6",
    description="Performance curve generator for building energy simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="building energy simulation performance curves",
    url="https://github.com/pnnl/copper",
    author="Jérémy Lerond",
    author_email="jeremy.lerond@pnnl.gov",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "statsmodels",
        "CoolProp",
        "scipy",
        "click",
    ],
    entry_points={
        "console_scripts": ["copper=copper.cli:cli"],
    },
    include_package_data=True,
    package_data={"copper": ["copper/lib/chiller_curves.json"]},
)
