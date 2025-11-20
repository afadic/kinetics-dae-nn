# kinetics-dae-nn
I am revisiting my PhD work on surrogate modeling with Neural Networks. Some of this work has been presented at conferences:
- A. Fadic Acceleration of detailed chemistry with neural networks: Canadian chemical engineering conference, Edmonton, October 22-25, 2017
- A. Fadic Detailed chemistry acceleration implemented into commercial CFD code. Modegat III, Bad Herrenalb, Germany, Sept 2017

The base case scenario is the Krahnert mechanism used for ammonia oxidation. These days this qualifies as a Physics Informed Neural Network.

One of the main objectives is to study the potential performance improvements and accuracy of surrogate models.

Krahnert.py has the mechanism encoded. The system is a DAE and it is solved using Sundials (IDA solver) compiled from source code. 
Validations are performed against data from the original paper. 

One of the comparisons done is the time savings using the analytically computed jacobian versus the finite-difference implemented one. This comparison is useful for baseline performance establishment.

Figure 6.6 of my thesis was replicated successfully (showing the adsoprtion dynamics, which plays a role in the time/spatial discretization for LES turbulence, another subject of my thesis)

If any of my work is useful to you, I am happy to hear from you.

Anton

