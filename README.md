# TQC

Source codes for implementation of the Kitaev Honeycomb model and related calculations.

This package was developed for research published in [arXiv:2003.07280](https://arxiv.org/abs/2003.07280) . 
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

- Use the docker file [docker_tqc](https://github.com/mrnp95/TQC/blob/master/docker_tqc).

## Documentation

- **show.py**

    - a utility script to show .log file generated by netket, either gui or cui 
    - see more usages with ./show.log --help

- **honeycomb.py**

    - some utility functions which will be used in many other codes. It includes, get_graph(), get_hamiltonian(), exact_diag(), gs_energy(), rbm(), measure() and flip_spins().
    - get_graph() is the most useful one. It generates an arbitary size honeycomb graph.
    - get_hamiltonian() uses get_graph() to get the Hamiltonian for an arbitary size honeycomb model.
    - after creating hamiltonian(by get_hamiltonian) and machine(derictly use netket), pass them into rbm() to train it.
    - exact_diag() gets energy level by directly diagnolize. gs_energy() gets gs energy by the analytic formula.
    - measure() is actually a lr=0 rbm().
    - flip_spins() manages to flip a spin by modifing parameters of an rbm.
    - This script is supposed to only be imported and used in other scripts. See contents of calc_gs.py as an example.

- **calc_gs.py**
    - use honeycomb.py to calc gs.
    
    ### Examples in Jupyeter notebooks

    Some basic examples of using built-in functions are demonstrated in [examples/example.ipynb](examples/example.ipynb).

In order to cite this work, use the following: 

### bibtex

	```
	@misc{noorm2020restricted,
		title={Restricted Boltzmann machine representation for the groundstate and excited states of Kitaev Honeycomb model},
		author={Mohammadreza Noormandipour and Youran Sun and Babak Haghighat},
		year={2020},
		eprint={2003.07280},
		archivePrefix={arXiv},
		primaryClass={cond-mat.dis-nn}
	}
	```

