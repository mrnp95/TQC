# TQC

Source codes for implementation of the Kitaev Honeycomb model and related calculations.

This package was developed for research published in arXiv:20... . In order to cite this work, use the following: 

### bibtex


## Installation

#### Conda

- Build the conda environment using `environment.yml` and activate it:
    ```
    conda env create -f environment.yml
    conda activate tqc
    ```
- Install [mpich](https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi) and then `netket`:
    ```
    pip install netket
    ```
#### Docker

- Use the docker file 'docker_tqc'
	```
	docker
	```

## Documentation

- **show.py**

    - a utility script to show .log file generated by netket, either gui or cui 
    - see more usages with ./show.log --help

- **honeycomb.py**

    - some utility functions which will be used in many other codes. It includes, get_graph_reza(), get_hamiltonian(), exact_diag(), gs_energy_babak(), rbm(), measure() and flip_spins()
    - you need to create machine yourself and pass it to rbm() to train it
    - This script is supposed to only be imported and used in other scripts. One of using demo of this script can be found in calc_gs.py

    ### Examples in Jupyeter notebooks

    Some basic examples of using built-in functions are demonstrated in [examples/example.ipynb](examples/example.ipynb).


