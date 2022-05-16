from setuptools import setup, find_packages

setup(
    name="copper",
    version="0.0.1",
    description="Performance curves generator for building energy simulation.",
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
