#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creditrisiko hjemmeopgaver
"""
#Hjemmeopgave 1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import CreditRisk.Merton as m

V = 130
Ds = 50
Dj = 50
r = 0.01
sigma = 0.25
T=1

Vs = np.arange(0,200,0.01)
Dlist = np.array((Ds,Dj))

m.merton(Vs, Ds, r, sigma, T)
m.merton(Vs, Dj, r, sigma, T)

prices = m.mertonMultiple(Vs, Dlist, r, sigma, T)


sns.set_theme()
plt.plot(prices)
plt.title("bond view")
plt.legend(prices.columns,loc="upper left")

plt.tight_layout()
    
plt.show()