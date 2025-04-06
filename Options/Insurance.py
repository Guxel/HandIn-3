#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio insurance for BS model with a put option and portfolio weight a
"""
from .base import claim
import numpy as np
import scipy

class insurance(claim):
    
    def payoff(self,**v):
        """
        Required
        ----------
        S : Price of underlying
        K : Strike Price
        
        
        Returns
        -------
        Payoff of option

        """
        if not "A" in v:
            g = v['A0'] / np.power(v['S0'],v['a']) * np.exp((v['r']+v['a']*(v['sigma']**2)/2)*(1-v['a'])*v['t'])
            v['A'] = g * np.power(v['S'],v['a'])
        return np.maximum(v['K']-v['A'],0)
    
    def price(self,**v):
        """
        Required
        ----------
        S : Price of underlying
        K : Strike Price
        r : Constant Interest Rate
        sigma : Constant Volatility
        T : Time till Maturity (NOT maturity date)
        
        
        Returns
        -------
        Market value of option

        """
        with np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
            if not "A" in v:
                g = v['A0'] / np.power(v['S0'],v['a']) * np.exp((v['r']+v['a']*(v['sigma']**2)/2)*(1-v['a'])*v['t'])
                v['A'] = g * np.power(v['S'],v['a'])
            d1 = (np.log(v['A'] / v['K']) + (v['r'] + 0.5 * (v['sigma']*v['a']) ** 2) * v['T']) / (v['a']*v['sigma'] * np.sqrt(v['T']))
            d2 = d1 - v['a']*v['sigma'] * np.sqrt(v['T'])
           
            opt = - v['A'] * scipy.stats.norm.cdf(-d1) + v['K'] * np.exp(-v['r'] *(v['T'])) * scipy.stats.norm.cdf(-d2)
            return  np.where(v['T']==0,self.payoff(**v),opt)
    
    def delta(self,**v):
        with np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
            if not "A" in v:
                g = v['A0'] / np.power(v['S0'],v['a']) * np.exp((v['r']+v['a']*(v['sigma']**2)/2)*(1-v['a'])*v['t'])
                v['A'] = g * np.power(v['S'],v['a'])
            d1 = (np.log(v['A'] / v['K']) + (v['r'] + 0.5 * (v['sigma']*v['a']) ** 2) * v['T']) / (v['a']*v['sigma'] * np.sqrt(v['T']))
            delta = - v['a'] * v['A'] / v['S'] * scipy.stats.norm.cdf(-d1) 
            return  np.where(v['T']==0,0,delta)
        
    def atm(self,**v):
        g = v['A0'] / np.power(v['S0'],v['a']) * np.exp((v['r']+v['a']*(v['sigma']**2)/2)*(1-v['a'])*v['t'])
        return np.power(v['K']/g,1/v['a'])
        
    def payoffD(self,**v):
        g = v['A0'] / np.power(v['S0'],v['a']) * np.exp((v['r']+v['a']*(v['sigma']**2)/2)*(1-v['a'])*v['t'])
        Satm = np.power(v['K']/g,1/v['a'])
        return - v['a'] * g * np.power(v['S'],v['a']-1) * np.where(v['S']<Satm,1,0)
    
    def payoffDD(self,**v):
        g = v['A0'] / np.power(v['S0'],v['a']) * np.exp((v['r']+v['a']*(v['sigma']**2)/2)*(1-v['a'])*v['t'])
        Satm = np.power(v['K']/g,1/v['a'])
        return (v['a']-v['a'] ** 2) * g * np.power(v['S'],v['a']-2)* np.where(v['S']<Satm,1,0) + v['a'] * g * np.power(v['S'],v['a']-1) * np.where(v['S']==Satm,1,0)


if __name__ == '__main__':
    insuranceOpt = insurance()
    r = 0.02
    std = 0.2
    A = S = 1
    T=30
    K = np.exp(r*T)
    a = 0.5
    insuranceOpt.price(r=r,sigma=std,A=A,T=T,K=K,a=a)
    insuranceOpt.price(r=r,sigma=std,A0=A,S0=S,S=S,T=T,t=0,K=K,a=a)
    insuranceOpt.delta(r=r,sigma=std,A=A,S=S,T=T,K=K,a=a)
    insuranceOpt.delta(r=r,sigma=std,A0=A,S0=S,S=S,T=T,t=0,K=K,a=a)
    pass
