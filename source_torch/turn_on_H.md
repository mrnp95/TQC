For Kiteav's honeycomb model, set K to nonzero is equivalent to turn on a background H field. In this case, the model is no longer analytically solvable. However, for the RBM approach, there is no difference whether K is zero or not. 

Take the 3x3 lattice, for example, set K=0.2 (where J is 1), the GS energy is -17.5260 with exact diagonalization. This K value stands for a medium H field as the GS energy without K is -14.2915. Using RBM, with the same number of parameters and training procedures for the K=0 system, we can get -17.393.

However, the energy for 1st excited state is still not good. The exact diagonalized 1st excited state energy is -16.5689, while RBM can only get -15.3399.

Item          |GS with K=0|GS with K=0.2|1st with K=0.2
:------------:|:---------:|:-----------:|:------------:
Exact Energies|-14.2915   |-17.5260     |-16.5689
RBM Energies  |           |-17.393      |-15.357
