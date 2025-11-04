# Bruggeman: Analytical Solutions of Geohydrological Problems

This repository contains implementations of Bruggeman's analytical solutions in Python.

The philosphy behind this repository is to collect implementations of
analytical solutions so they are readily available for use in projects or for
benchmarking other computations or models.

Very much a work in progress. Any contributions are welcome.

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

## Contributing

Any contributions are welcome! For the best results please follow the guidelines below:

- The analytical solutions are generally stored in the `bruggeman/*.py` files. Please
  select the appropriate python file for your solution. E.g. 1D flow solutions should be
  stored `flow1d.py`. Create a new file if your solution does not fit into the currently
  available files.
- Create a notebook showcasing your solution. Put the notebook in `docs/examples/`.
- Add the notebook under the appropriate section in `docs/examples/index.rst`. Follow
  the examples in the file to add your notebook. Don't forget to add your notebook to the
  table of contents as well.
- Consider decorating your function with latexify by importing
  `from bruggeman.general import latexify_function`. Follow the examples already present
  in the python files. This will allow you to render your formulas with LateX in a notebook.
