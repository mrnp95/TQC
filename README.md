# TQC
Source codes for implementation of the Kitaev Honeycomb model!

## Codes Description

* exact_diag_for_reza.py
    * generate a graph for Kitaev's honeycomb model
    * generate its Hilbert Space and Hamiltonian based on the graph
    * use netket.exact.lanczos_ed() to exact diagonalize the Hamilyonian numerically
* calc_gs.py
    * generate a graph for Kitaev's honeycomb model
    * generate its Hilbert Space and Hamiltonian based on the graph
    * create a RBM mechine, a sampler, an optimizer and a Vmc 
    * calculate ground state
* test_flip_and_measure.py
    * generate a simple testing graph and its Hamiltonian
    * create a RBM mechine for it and initialize the mechine by random parameters
    * try measuring in netket
    * try to flip a spin and check

## Results
