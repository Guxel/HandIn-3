#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bachelier Call option
"""
from .base import claim
import numpy as np
from Helpers import stats 

class bachelierCall(claim):
    
    def payoff(self,**v):
        return np.maximum(v['S']-v['K'],0)
    
    def price(self,**v):
        with stats.work_precision(self.precision), np.errstate(divide='ignore', invalid='ignore'): #for T=0
            d = (v['S'] - v['K']) / (v['sigma'] * stats.sqrt(v['T']))
            price = (v['S']-v['K']) * stats.norm_cdf(d) \
                  + (v['sigma'] * stats.sqrt(v['T'])) * stats.norm_pdf(d)
            return  np.where(v['T']==0,self.payoff(**v),price)
    


if __name__ == '__main__':
    opt = bachelierCall()
    opt.price(S=100,K=5,sigma=15,T=0.25)
    