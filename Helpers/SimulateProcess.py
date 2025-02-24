#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate hedge
"""

import numpy as np
import scipy


def payoff(ST,K):
    return np.maximum(0,ST-K)

def simulate_brownian(delta,n,m,p=1):
    brownian = np.zeros([p,n+1,m])
    brownian[:,1:] = np.random.normal(0,np.sqrt(delta),[p,n,m])
    return np.cumsum(brownian, axis = 1)

def Simulate_GBM(T,delta,S0,mu,sigma,m=1):
    n = int(T/delta)
    brownian = np.zeros([n+1,m])
    brownian[1:] = np.random.normal(0,np.sqrt(delta),[n,m])
    brownian = np.cumsum(brownian, axis = 0)
    
    deltaT = np.arange(0,m+1).reshape(-1, 1) * delta
    
    return S0 * np.exp((mu-0.5 * (sigma**2) ) * deltaT + sigma * brownian)

def Simulate_GBM_End(T,S0,mu,sigma,n=1):
    """
    Returns last values only
    """
    brownian = np.random.normal(0,np.sqrt(T),n)
    return S0 * np.exp((mu-0.5 * (sigma**2) )*T + sigma *brownian)

def Simulate_GBM_Dyamics(T,delta,S0,mu,sigma,n=1):
    """
    Uses dynamics for simulation, not accurate for large delta
    """
    m = int(T/delta)
    path = np.zeros([m+1,n])
    path[0] = S0
    brownian = np.random.normal(0,1,[m,n])
    
    add = 1+mu*delta
    vol = sigma*np.sqrt(delta)
    
    for i in range(1,m+1):
        path[i] = path[i-1]*(add + brownian[i-1] * vol)
    return path

def simulate_options(T,n,S0,r,sigma,K):
    payoffs = np.zeros((n,2))
    payoffs[:,0] = Simulate_GBM_End(T,S0,r,sigma,n)
    payoffs[:,1] = payoff(payoffs[:,0],K)
    return payoffs.T
    

def blackScholesCall(T, S0, r, sigma, K):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)
    return call_price

    

     
if __name__ == '__main__':
    T = 2
    delta = 1/10000
    S0=1
    mu = 0.08
    r = 0.03
    sigma=0.2
    K=1.1
    
    proc = Simulate_GBM(T,delta,S0,r,sigma,10000)
    end = Simulate_GBM_End(T,S0,r,sigma,100000)
    proc[-1].mean() - end.mean()
    
    opt = simulate_options(T,10000,S0,r,sigma,K)
    np.mean(opt[1])*np.exp(-r*T) - blackScholesCall(T,S0,r,sigma,K)
    
    
    
    
    
    
    
    
    
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme()
    plt.plot(np.arange(0,int(T/delta+1))*delta,proc[:,0:10])
    plt.title("Simuleret Process")
    
    plt.tight_layout()
    
    plt.show()
    
    
    sns.set_theme()
    plt.plot(opt[0],opt[1], 'o')
    plt.title("Option Simulation")
    
    plt.tight_layout()
    
    plt.show()

