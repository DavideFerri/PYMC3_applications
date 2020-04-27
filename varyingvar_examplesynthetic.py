#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:33:38 2020

@author: davideferri
"""

import logging 
import numpy as np 
import pandas as pd
import scipy.stats as ss
import pymc3 as pm 
import arviz as az

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------  simulate data ----------------------------------------- # 

# set the random seed 
np.random.seed(123)
# set the number of draws 
N = 1000
# define the independent variable 
x = np.linspace(-50,50,N)
# define the true coefficients
alpha_true = 1 ; beta_true = 2 ; gamma_true = 1 ; delta_true = 0.5
# define the dependant variable
y = np.random.normal(loc = alpha_true + beta_true * x, scale = gamma_true + delta_true * np.abs(x))
# plot
fig, ax = plt.subplots()
ax.scatter(x,y)
ax.grid(True)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()


# -------------------- specify a probabilistic model for the data -------------- # 

with pm.Model() as vv_model:
    # specify the priors for the parameters
    alpha = pm.Normal("alpha", mu = 0, sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10)
    gamma = pm.HalfNormal("gamma", 10)
    delta = pm.HalfNormal("delta", 10)
    # specify the mean of the observations
    mu = pm.Deterministic("mu", alpha + beta * x)
    # specify the variance of the observations
    eps = pm.Deterministic("eps", gamma + delta * np.abs(x))
    # set the likelihood of observations
    y_obs = pm.Normal("y_obs", mu = mu, sd = eps, observed = y)
    # inference step 
    trace = pm.sample(1000, tune = 2000)
    print(az.summary(trace, var_names = ["alpha","beta","gamma","delta"]))