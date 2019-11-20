# TQC
Source codes for implementation of the Kitaev Honeycomb model!

## Codes Description

* show.py
    * a utility script to show .log file generated by netket, either gui or cui
    * see more usages with `./show.log --help`
* honeycomb.py
    * some utility functions which will be used in many other codes. It includes, get_graph_reza(), get_hamiltonian(), exact_diag(), gs_energy_babak(), rbm(), __measure()__ and __flip_spins()__
    * you need to create machine yourself and pass it to rbm() to train it
    * __flip_spins() is still being testing its stability. Be careful when using it__ 
    * This script is supposed to only be imported and used in other scripts. One of using demo of this script can be found in calc_gs.py
* exact_diag_for_reza.py
    * a script to show Reza that the exact diagonalize energy is different from Babak's formula
* test_flip_and_measure.py
    * test measure and flip technology by Ising model
* play_with_alpha.py
    * try how large of alpha is suitable, by a 5x5 lattice, 1kx10k trains and alpha 0.1,0.3,0.5,1,1.5,2,3,4,6,8,10.
* try_other_machines.py
    * try ffnn and compare it with rbm, from the perspective of time, effecient of parameters and, the most important, accuracy.

## Results

### Play with alpha: how many hidden nodes are suitable?

Study how number of hidden nodes, denoted by num_nh, influences accuracy and training time. Fix the __lattice size to 5x5__, train batch number to 1kx10k, optimizer to Sgd(learning_rate=0.01,decay_factor=1), sampler to MetropolisLocal. The exact ground state energy of 5x5 lattice is -39.3892. In the table below, -27.7808(59) means $-27.7808\pm0.0059$.

|alpha|num_nh|num_para|energy|time(s)|notes|
|:---:|:----:|:------:|:----:|:--:|:---:|
|0.1|  5| 305|-27.7808(59)|171.6|
|0.3| 15| 815|-33.4312(37)|455.0|
|0.5| 25|1325|-36.6024(22)|711.9|
|1.0| 50|2600|-37.2464(13)|1549|
|1.5| 75|3875|-37.1235(15)|2764|
|2.0|100|5150|-37.3054(14)|5611|
|3.0|150|7700|-37.3037(14)|8415|
|4.0|200|10250|-37.1007(16)|11204|
|6.0|300|15350|-37.2989(14)|18167|
|8.0|400|20450|-37.1710(15)|26000|
|10.0|500|25550|-37.1946(15)|32056|

* __2 is the most suitable value for alpha__. If it is too low, the fitting performace is not good; too high overfit will happen.
* The training time is approximately proportional to the number of parameters. __For each additional parameter, the training time is increased by about 1 second.__
* __The result of RBM is still far away from the analytic result.__

### The performance of FFNN
Below is the result of FFNN with one layer. As the structure of ffnn is similar to that of a no-visible-bias RBM(reffered as novb RBM below), the performance of one-layer FFNN is compared to that of no-visible-bias RBM.

|alpha|num_nh|num_para|ffnn_energy|novb_rbm_energy|ffnn_time(s)|novb_rbm_time|notes|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1.0| 50|2550|-37.2401(15)|-37.1532(15)|2783|2754|
|2.0|100|5100|-37.2958(14)|-36.8954(18)|4591|5476|
|3.0|150|7650|-37.3007(14)|-37.2191(15)|6889|8331|

* FFNN trains faster than RBM and performs better than RBM with no bias.
* But FFNN's result again far away from the analytic one.

Deep FFNNs with 2 and 3 layers are also tried. But some bugs seem happen, the energies stay at -25. It will be studied later.
