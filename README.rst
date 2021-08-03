ncem benchmarks
=====================================================================

Collection of scripts and notebooks to create panels in publication.
This repository contains grid search preparation, execution and evaluation code used in the original ncem_ publication_.

Next to python and shell scripts for grid searches and jupyter notebooks for results evaluation, this repository contains shallow infrastructure for defining hyperparameters in grid searches under `ncem_benchmarks/`.
Install this package via `pip install -e .` into a python environment with an existing ncem installation to make this infrastructure available to the grid search scripts defined in this repository.

Before running grid searches, prepare the data as described in `notebooks/data_preparation/`.
Grid searches and production model training can be run using the scripts as described in `scripts/grid_searches/`.


.. _ncem: https://ncem.readthedocs.io
.. _publication :