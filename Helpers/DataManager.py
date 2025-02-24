#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data manager
"""

import pandas as pd

dataPath = r"/Users/gustavdyhr/Desktop/Studie/8 Semester/Python/Data/"

def readCR_Excel(name, extension = ".xlsx"):
    return pd.read_excel(dataPath + "CreditRisk/" + name + extension,engine="openpyxl")

if __name__ == '__main__': 
    nf = readCR_Excel('netflix')