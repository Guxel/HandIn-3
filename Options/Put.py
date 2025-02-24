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
        return np.maximum(v['K']-v['S'],0)
    
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
            d1 = (np.log(v['S'] / v['K']) + (v['r'] + 0.5 * v['sigma'] ** 2) * v['T']) / (v['sigma'] * np.sqrt(v['T']))
            d2 = d1 - v['sigma'] * np.sqrt(v['T'])
           
            opt = - v['S'] * scipy.stats.norm.cdf(-d1) + v['K'] * np.exp(-v['r'] * v['T']) * scipy.stats.norm.cdf(-d2)
            return  np.where(v['T']==0,self.payoff(self,**v),opt)


if __name__ == '__main__':
    pass
