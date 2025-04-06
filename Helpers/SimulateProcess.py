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

def Simulate_GBM(T,delta,S0,mu,sigma,m=1,antihetic = False):
    n = int(T/delta)
    brownian = np.zeros([n+1,m])
    if antihetic:
        brownian[1:,0:int(m/2)] = np.random.normal(0,np.sqrt(delta),[n,int(m/2)])
        brownian[1:,int(m/2):int(m/2)*2] = -brownian[1:,0:int(m/2)]
        if m & 0x1:
            brownian[1:,-1] = np.random.normal(0,np.sqrt(delta),n)
    else:
        brownian[1:] = np.random.normal(0,np.sqrt(delta),[n,m])
    brownian = np.cumsum(brownian, axis = 0)
    
    deltaT = np.arange(0,n+1).reshape(-1, 1) * delta
    
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

def simulateHestonDynamics(T,dt,S0,mu,V0,theta,kappa,epsilon,rho,n=1):
    """
    Uses dynamics for simulation, not accurate for large dt
    """
    m = int(T/dt)
    path = np.zeros([m+1,n,2])
    path[0,:,0] = S0
    path[0,:,1] = V0
    
    stockBrownian = np.sqrt(dt)*np.random.normal(0,1,[m,n])
    volMovements = epsilon*(rho*stockBrownian +np.sqrt(1-rho**2)*np.sqrt(dt)*np.random.normal(0,1,[m,n]))
    
    driftStock = mu*dt
    kapdt = kappa * dt
    driftVol = theta*kapdt
    
    
    for i in range(1,m+1):
        vol = np.sqrt(path[i-1,:,1])
        path[i,:,0] = np.exp(
                    np.log(path[i-1,:,0])
                    +driftStock
                    -0.5*path[i-1,:,1]*dt
                    +vol*stockBrownian[i-1]
                    )
        path[i,:,1] = np.maximum(
                                path[i-1,:,1]
                                +driftVol-path[i-1,:,1]*kapdt
                                +vol*volMovements[i-1]
                                ,0)
    return path

def simulateHestonDynamicsBatched(T, dt, S0, mu, V0, theta, kappa, epsilon, rho, n=1, batchSize=10000,float32 = True):
    """
    Simulates the Heston dynamics in batches to avoid memory issues when n is large and dt is small.
    """
    m = int(T / dt)
    results = []
    batches = int(np.ceil(m / batchSize))
    batchN =  int(np.ceil(n/batches))
    print(f"Doing {batches} batches")
    
    driftStock = mu * dt
    kapdt = kappa * dt
    driftVol = theta * kapdt
    
    for b in range(batches):
        print(f"Starting batch {b+1}")
        n_batch = batchN if (b + 1) * batchN <= n else n - b * batchN
        
        if float32:
            path = np.zeros([m + 1, n_batch, 2],dtype=np.float32)
        else:
            path = np.zeros([m + 1, n_batch, 2])
        path[0, :, 0] = S0
        path[0, :, 1] = V0
        
        stockBrownian = np.sqrt(dt) * np.random.normal(0, 1, [m, n_batch])
        volMovements = epsilon * (
            rho * stockBrownian +
            np.sqrt(1 - rho ** 2) * np.sqrt(dt) * np.random.normal(0, 1, [m, n_batch])
        )
        
        
        for i in range(1, m + 1):
            vol = np.sqrt(path[i - 1, :, 1])
            path[i, :, 0] = np.exp(
                np.log(path[i - 1, :, 0])
                + driftStock
                - 0.5 * path[i - 1, :, 1] * dt
                + vol * stockBrownian[i - 1]
            )
            path[i, :, 1] = np.maximum(
                path[i - 1, :, 1]
                + driftVol - path[i - 1, :, 1] * kapdt
                + vol * volMovements[i - 1],
                0
            )
        
        results.append(path)
    
    return np.concatenate(results, axis=1)



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

