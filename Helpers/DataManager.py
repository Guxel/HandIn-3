#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data manager
"""

import pandas as pd
import numpy as np

dataPath = r"/Users/gustavdyhr/Desktop/Studie/8 Semester/Python/Data/"

def readCR_Excel(name, extension = ".xlsx"):
    return pd.read_excel(dataPath + "CreditRisk/" + name + extension,engine="openpyxl")

def shift(xs, n, axis = 0):
    e = np.empty_like(xs)
    if axis == 1:
        e = e.T
        xs = xs.T
        
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
        
    if axis == 1:
        return e.T
    else:
        return e


if __name__ == '__main__': 
    nf = readCR_Excel('netflix')