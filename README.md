# An approximate diffusion process for environmental stochasticity in infectious disease transmission modelling

This repository contains code that demonstrate the application of approximate diffusion processes for environmental stochasticity, supporting the above paper. In particular this is the code to fit the influenza data used in the paper while using a SDE and its approximation to represent the environmental stochasticity in a SIR model. 

## Dependencies
Generic scientific Python stack: `numpy`, `scipy`, `matplotlib`, `sklearn`, `seaborn`, `joblib`, and `arviz` (0.4.1).

The particle filter is implemented using `Jax`, to benefit from JIT compilation (and probable GPU usage down the line). Install `numpyro` which will also install `Jax`

To install `NumPyro` read the following:
http://num.pyro.ai/en/stable/getting_started.html#installation 