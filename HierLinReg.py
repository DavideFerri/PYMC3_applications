#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:32:05 2020

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

# --------------------- generate synthetic data -------------------------- # 

# get the number of observations by group
N = 20 
# get the number of groups 
M = 8
# define the index array; all group have N observations but the last (only 1)
idx = np.repeat(range(M-1),N)
idx = np.append(idx,7)
log.info("The index list is: %s",idx)
# set a random seed 
np.random.seed(314)

# define the real coefficients
alpha_real = ss.norm.rvs(loc=2.5,scale=0.5,size=M)
log.info("The alpha real is: %s", alpha_real)
beta_real = np.random.beta(6,1,size=M)
log.info("The beta real is: %s", beta_real)
eps_real = np.random.normal(0,0.5,size=len(idx))

# set the independent variable
x_m = np.random.normal(10,1,len(idx))
# set the dependent variable
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real
# plot the true data
fig,ax = plt.subplots(2,4, figsize = (10,5), sharex = True, sharey = True)
ax = np.ravel(ax)
# initialize j and k
j, k = 0, N
for i in range(M):
    # scatter the data
    ax[i].scatter(x_m[j:k],y_m[j:k])
    # set the x label
    ax[i].set_xlabel(f"x_{i}")
    # set the y label
    ax[i].set_ylabel(f"y_{i}",rotation = 0, labelpad = 15)
    # set the x axis limit
    ax[i].set_xlim(6,15)
    # set the y axis limit
    ax[i].set_ylim(7,17)
    # update j,k 
    j += N
    k += N
plt.tight_layout()
plt.show()
    
    
    
    
    