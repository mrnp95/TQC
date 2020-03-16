# TQC
Source codes for implementation of the Kitaev Honeycomb model and related calculations.

This package was developed for research published in arXiv:20... . In order to cite this work, use the following: 

### bibtex


## Installing:

## Documentation:


    *show.py\n
        *a utility script to show .log file generated by netket, either gui or cui
        see more usages with ./show.log --help
    honeycomb.py
        some utility functions which will be used in many other codes. It includes, get_graph_reza(), get_hamiltonian(), exact_diag(), gs_energy_babak(), rbm(), measure() and flip_spins()
        you need to create machine yourself and pass it to rbm() to train it
        This script is supposed to only be imported and used in other scripts. One of using demo of this script can be found in calc_gs.py

### Examples in Jupyeter notebooks:


