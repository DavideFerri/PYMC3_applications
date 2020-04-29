#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:51:18 2020

@author: davideferri
"""

import numpy as np 
import pandas as pd 
import scipy.stats as ss
import pymc3 as pm 
import arviz as az
import logging
import matplotlib.pyplot as plt 
import seaborn as sns

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')
# ---------------------- import the data ----------------------------- 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_length", data = iris, jitter = True)

# ---------------------- transformations ------------------------ #

# keep only setosa and versicolor
iris = iris[(iris["species"] == "setosa")|(iris["species"] == "versicolor")]
# set the dependant variable
y_0 = pd.Categorical(iris["species"]).codes
# set the independent variable
x_0 = iris["petal_length"].values
# center the independent variable 
x_n = x_0 - x_0.mean()

# --------------------- specify the probabilistic model --------- # 

with pm.Model() as log_model:
    # set the priors 
    alpha = pm.Normal("alpha",mu = 0,sd = 10)
    beta = pm.Normal("beta", mu = 0, sd = 10)
    # set the bernoulli parameter
    theta = pm.Deterministic("theta",pm.math.sigmoid(alpha + beta * x_n))
    # set the decision boundary 
    db = pm.Deterministic("db", -alpha/beta)
    # set the likelihood 
    y_obs = pm.Bernoulli("y_obs",p = theta, observed = y_0)
    # inference step
    trace = pm.sample(1000)