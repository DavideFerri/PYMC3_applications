#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:19:14 2020

@author: davideferri
"""

import logging
import pandas as pd
import numpy as np 
import scipy.stats as ss 
import arviz as az
import pymc3 as pm 
import seaborn as sns

# --------------------------------- import the data --------------------------------------------- # 

data = pd.read_csv('./data/tips.csv')
log.info("The tips data tail is as follows: %s", data.tail())
# plot the data by day
sns.violinplot(x = "day", y = "tip", data = data)

# --------------------------------- set variables -------------------------------------- #

# get the tips
tips = data.tip.values
# get the days and turn them into categories 0,1,2,3
days = pd.Categorical(data["day"],categories = ["Thur","Fri","Sat","Sun"]).codes
# get a variable equal to the number of categories 
cat_number = len(np.unique(days))

# ------------------------- specify the probabilistic model ------------------------ # 

with pm.Model() as model:
    # set the prior for the location parameter
    mu = pm.Normal("mu", mu = 0, sd = 10, shape = cat_number)
    # set the prior for the scale parameter
    sigma = pm.HalfNormal("sigma", sd = 10, shape = cat_number)
    # specify the likelihood of the data
    obs = pm.Normal("obs", mu = mu[days], sigma = sigma[days], observed = tips)
    # inference step 
    trace = pm.sample(1000)
    
# ------------------------- analyse the posterior ------------------------------- # 
    
with model: 
    # get the MAP estimates
    map_estimates = pm.find_MAP()
    log.info("The map estimates are: %s", map_estimates)
    # print the trace
    log.info("The summary of the mu trace with shape %s is: %s",trace["mu"].shape,trace["mu"])
    log.info("The summary of the sigma trace with shape %s is: %s",trace["sigma"].shape,trace["sigma"])
    # print a summary of the results
    log.info("The summary of the posterior is : %s", az.summary(trace))
    az.plot_trace(trace)


