#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:29:41 2020

@author: davideferri
"""

import logging 
import numpy as np 
import pandas as pd
import scipy.stats as ss
import pymc3 as pm 
import arviz as az
import matplotlib.pyplot as plt 

# ---------------- import the iris data ----------------------------- # 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_width", data = iris, jitter = True)