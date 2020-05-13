#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:18:36 2020

@author: davideferri
"""

import logging 
import numpy as np 
import pandas as pd
import scipy.stats as ss
import pymc3 as pm 
import arviz as az
import matplotlib.pyplot as plt 

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# ---------------- import the iris data ----------------------------- # 

iris = pd.read_csv('./data/Iris.csv')
log.info("The head of the Iris dataset is: %s", iris.head())
# plot the three species vs petal lenght
sns.stripplot(x ="species", y = "petal_width", data = iris, jitter = True)

# ---------------- transformations ---------------------------------- #

iris = iris.query("species == ('setosa','versicolor')")
y_0 = pd.Categorical(iris["species"]).codes
x_0 = iris["petal_length"].values
y_0 = np.concatenate((y_0,np.ones(6,dtype = int)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
# center the data
x_c = x_0 - x_0.mean()
plt.figure()
plt.plot(x_c,y_0,"o",color = "k")
plt.show()