# Bruggeman: Analytical Solutions of Geohydrological Problems

This repository contains implementations of Bruggeman's analytical solutions in Python.

The philosphy behind this repository is to collect implementations of
analytical solutions so they are readily available for use in projects or for
benchmarking other computations or models.

Very much a work in progress.

## Installation

Normal install:

`pip install bruggeman`

Development install:

`pip install -e .`

## Documentation

The documentation is available [here](https://bruggeman.readthedocs.io/en/latest/index.html).

To build the documentation locally:

1. Install the optional documentation dependencies `pip install bruggeman[docs]`
(or `pip install -e ".[docs]"`).
2. Navigate to `docs/`
3. Enter the command `make html`.
4. The documenation is contained in the `docs/_build` folder. Open `index.html` in
your browser to view the documentation.
