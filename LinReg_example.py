#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:57:11 2020

@author: davideferri
"""

import logging 
import numpy as np 
import pandas as pd 
import scipy.stats as ss
import matplotlib.pyplot as plt
import pymc3 as pm 
import arviz as az

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')
# ---------------------- generate synthetic data --------------------------------------- # 

# number of obs
N = 101
# set the random seed
np.random.seed(123)
# set the independent variable data
x = np.linspace(0,100,N)
# set the value of the true coefficients 
b = 0.35 ; s = 5 ; alpha = 1
# get the errors
err = ss.norm.rvs(loc = 0,scale = s, size = N)
# get the y's
y = alpha + b * x + err
# plot the true data
fig,ax = plt.subplots(1,2,figsize = (8,4))
ax[0].scatter(x,y)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
az.plot_kde(y, ax = ax[1])
ax[1].set_xlabel("y")
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].tick_params(axis='both', which='minor', labelsize=10)
plt.tight_layout()
plt.show()

# ---------------------- specify the pymc3 model ---------------------------------- # 

with pm.Model() as linear_model:
    # set the priors
    beta = pm.Normal("beta",0,10)
    alpha = pm.Normal("alpha",0,10)
    sigma = pm.HalfNormal("sigma",10)
    # set the likelihood of the observations
    obs = pm.Normal("obs", mu = alpha + beta * x, sigma = sigma, observed = y)
    # inference step 
    trace = pm.sample(2000,tune = 1000)
    
# # ------------------- analyse the posterior ------------------------------------- # 
    
with linear_model:
    log.info("the trace summary is: %s", az.summary(trace))
    #plt the results
    az.plot_joint(trace, kind = "kde", var_names = ["beta","alpha"])
    az.plot_trace(trace)
    az.plot_posterior(trace, var_names = ["alpha","beta"], rope = [-0.05,0.05], credible_interval = 0.9)
    