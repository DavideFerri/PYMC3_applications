#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:44:23 2020

@author: davideferri
"""

import numpy as np 
import pandas as pd 
import scipy.stats as ss 
import matplotlib.pyplot as plt 
import arviz as az
import pymc3 as pm
import logging 

# initialize the logger
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(name)s - %(levelname)s - %(message)s')

# --------------------------- import the data ----------------------------------- # 

data = pd.read_csv('https://stats.idre.ucla.edu/stat/data/fish.csv')