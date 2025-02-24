#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard merton lib
"""


import numpy as np
import pandas as pd
import scipy
from Options.Call import call

def merton(V, D, r, sigma, T):
    S = call.price(call,S=V,K=D,r=r,sigma=sigma,T=T)
    B = V - S
    return S, B

def mertonMultiple(V, D : np.ndarray, r : float, sigma : float, T : float):
    prices = pd.DataFrame(index = V)
    otherDebt = 0
    otherPrices = 0
    i = 1
    for d in D:
        S = call.price(call,S=V,K=d + otherDebt,r=r,sigma=sigma,T=T)
        B = V - S - otherPrices
        prices['Bond ' + str(i)] = B
        
        otherDebt += d
        otherPrices += B
        i += 1
    prices['Stock'] = S
    return prices


def bondYield(B, D, T):
    return - np.log(B/D) * (T ** -1)

def creditSpread(y, r):
    return y - r

def bondCreditSpread(B, D, T, r):
    return creditSpread(bondYield(B, D, T), r)
    
def vassalouXing(S : np.ndarray,D,T,r,dt = 1/255,sigmaGuess = 0.4,CONVERGENCE = 1E-07,MAX_ITERATIONS = 1000):
    N = S.size - 1 #Amount of movements
    
    for n in range(0,MAX_ITERATIONS):
        V = call.invertS(call,price = S,K=D,r=r,sigma=sigmaGuess,T=T)
        xi = 1 / (N*dt) * np.log(V[-1]/V[0])
        
        sigmaLast = sigmaGuess
        sigmaGuess = np.sqrt(1 / (N*dt) * np.sum(np.power(np.log(V[1:]/V[:-1])-xi*dt,2)))
        print(sigmaGuess)
        if abs(sigmaGuess-sigmaLast)<CONVERGENCE:
            return V, sigmaGuess
        
    print("Vassalou-Xing warning: Convergence not reached")
    return V, sigmaGuess

def MertonCouponsMonteCarlo(V0, D : np.ndarray, r : float, sigma : float, T : float, asset_sales : bool, n = 100000):
    sim = np.random.normal(0,1,n)
    simCompany = V0 * np.exp((r - 0.5 * sigma ** 2) * T + np.sqrt(T) * sigma * sim)
    for i in range(len(D),1):
        CurrentDebt = D[i-1]
        NextDebt = D[i]
        if asset_sales:
            simStock = call.price(call,S=simCompany,K=D[1],r=r,sigma=sigma,T=T)
        else:
            simStock = call.price(call,S=simCompany,K=D[1],r=r,sigma=sigma,T=T)


if __name__ == '__main__':
    import Helpers.DataManager as dm
    data = dm.readCR_Excel('netflix')
    r = data['1-year swap rate (percent)'] / 100
    dt = 1/12
    T = 1
    S = data['Stock price'] * data['Shares outstanding (milions)'] * 1000000
    D = (data['Current liabilities (millions)'] + 0.5 * data['Non-current liabilities (millions)']) * 1000000
    
    V, sigma = vassalouXing(S=S,D=D,T=T,r=r,dt=dt)
    
    