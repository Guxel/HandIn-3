#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary option of type
Payoff = 1 if S > K else 0 end
"""
from .base import claim
import numpy as np
import scipy

class binary(claim):
    
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
        return np.where(v['S']>v['K'],1,0)
    
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
            return np.where(v['T']==0,self.payoff(self,**v),np.exp(-v['r']*v['T'])*scipy.stats.norm.cdf( 
                    (np.log(v['S']/v['K'])+(v['r']-1/2*(v['sigma'] ** 2) )*v['T'])/(v['sigma']*np.sqrt(v['T']))
                ))

    
    def delta(self,**v):
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
        Delta of option

        """
        with np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
            return np.where(v['T']==0,0,np.exp(-v['r']*v['T'])*scipy.stats.norm.pdf(
                (np.log(v['S']/v['K'])+(v['r']-1/2*(v['sigma'] ** 2) )*v['T'])/(v['sigma']*np.sqrt(v['T'])))
                / (v['S']*v['sigma']*np.sqrt(v['T'])))


if __name__ == '__main__':
    opt = binary()
    opt.price(S=100,K=105,r=0.05,sigma=0.2,T=1)
    opt.delta(S=100,K=105,r=0.05,sigma=0.2,T=1)
    
    opt.price(S=np.array([100,100]),K=np.array([105,105]),r=np.array([0.05,0.05]),sigma=np.array([0.2,0.2]),T=np.array([1,2]))
    opt.delta(S=np.array([100,100]),K=np.array([105,105]),r=np.array([0.05,0.05]),sigma=np.array([0.2,0.2]),T=np.array([1,2]))
    0.01889073
    opt.price(S=100,K=105,r=0.05,sigma=0.2,T=np.array([1,2]))
    opt.price(S=100,K=105,r=np.array([0.05]),sigma=np.array([0.2,0.1]),T=np.array([1,2]))
    
