# An approximate diffusion process for environmental stochasticity in infectious disease transmission modelling

This repository contains code that demonstrate the application of approximate diffusion processes (SA) for environmental stochasticity, supporting the above paper. We have provided code and real/simulated data to reproduce the influenza modelling with SA/SDE. We have also included code and simulated data for the SIRS model. However, since the covid data is not publicly available we are only providing the code to fit the covid model. Access to the covid data can be made on request, see the `Data availability statement` in the paper. 

## Dependencies
Generic scientific Python stack: `numpy`, `scipy`, `matplotlib`, `pandas`, `sklearn`, `seaborn`, `joblib`.

The particle filter is implemented using `Jax`, to benefit from JIT compilation (and probable GPU usage down the line). Install `numpyro` which will also install `Jax`

To install `NumPyro` read the following:
http://num.pyro.ai/en/stable/getting_started.html#installation 


To wrapp the c++ code (needed to run the covid models) in python, the `pybind11` package must be installed: https://pybind11.readthedocs.io/en/latest/installing.html#include-with-pypi 

## Influenza model
To run the fitting process, SA with `15` basis functions and SDE `100` particles for the SMC:
 `SIR_example.py --iterations 100000 --burnin 50000 --thin 50 --n_bases 15 --n_particles 100`

To reproduce the MMD study, first run:
`vary_n_study.py` then run `plot_mmd.py`

## Nonstationary signal (with SIRS model)
To run the fitting process, SA with `20` basis functions:
 `SIRS_example.py --iterations 500000 --burnin 250000 --thin 250 --n_bases 2`

## COVID19 model
### Compile the c++ code
Go to `./models/COVID_CPP` directory and then compile by using `python setup.py build_ext -i`. Then rename the generated *.so files to `death_lik.so` and `seeiir_ode.so`.

### Do inference
Once you have access to the data, move these to the data directory. Then to run the model with random-walk stochasticity:
`COVID_rw_example.py` and with the Brownian motion approximation `COVID_bma_example.py`

Once MCMC is finished for both these, then run:
`plot_covid.py` to visualise the posterior predicitves.