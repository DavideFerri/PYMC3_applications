#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:44:55 2020

@author: davideferri
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss 
import pymc3 as pm 
import arviz as az

# --------------------------- define the 2nd Ascombes's quartet data ------------------------------------ # 

# define arrays of x and y values from the second Ascombe's quartet 
x = np.array([10,8,13,9,11,14,6,4,12,7,5])
y = np.array([9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74])
# plot the second Ascombe's quarter 
fig, ax = plt.subplots(figsize = (12,5))
ax.scatter(x,y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("The second Ascombe's quartet")
plt.tight_layout()
plt.show()
 