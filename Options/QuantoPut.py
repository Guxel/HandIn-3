#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quanto put option
"""
from .base import claim
import numpy as np
from Helpers import stats 
import scipy 


class quantoPut(claim):
    
    def payoff(self,**v):
        return v['Y']*np.maximum(v['K']-v['S'],0)
    
    def price(self,**v):
        if self.precision ==16:
            with np.errstate(divide='ignore', invalid='ignore'): #for T=0
                norm = np.dot(v['sigmaF'].T,v['sigmaF'])
                comp = np.dot(v['sigmaX'].T,v['sigmaF'])
                d1 = (np.log(v['S']/ v['K'])+(v['rF']-comp+1/2*norm)*v['T'])/(np.sqrt(norm*v['T']))
                d2 = d1 - np.sqrt(norm*v['T'])
                price = v['Y']*np.exp(-v['rD']*v['T'])*(v['K']*scipy.stats.norm.cdf(-d2)-np.exp((v['rF']-comp)*v['T'])*v['S']*scipy.stats.norm.cdf(-d1))
                return  np.where(v['T']==0,self.payoff(**v),price)
        
        else:
            with stats.work_precision(self.precision), np.errstate(divide='ignore', invalid='ignore'): #for T=0
                norm = np.dot(v['sigmaF'].T,v['sigmaF'])
                comp = np.dot(v['sigmaX'].T,v['sigmaF'])
                d1 = (stats.log(v['S']/ v['K'])+(v['rF']-comp+1/2*norm)*v['T'])/(stats.sqrt(norm*v['T']))
                d2 = d1 - stats.sqrt(norm*v['T'])
                price = v['Y']*stats.exp(-v['rD']*v['T'])*(v['K']*stats.norm_cdf(-d2)-stats.exp((v['rF']-comp)*v['T'])*v['S']*stats.norm_cdf(-d1))
                return  np.where(v['T']==0,self.payoff(**v),price)
        
    def delta(self,**v):
        if self.precision ==16:
            with np.errstate(divide='ignore', invalid='ignore'): #for T=0
                norm = np.dot(v['sigmaF'].T,v['sigmaF'])
                comp = np.dot(v['sigmaX'].T,v['sigmaF'])
                d1 = (np.log(v['S']/ v['K'])+(v['rF']-comp+1/2*norm)*v['T'])/(np.sqrt(norm*v['T']))
                return np.where(v['T']==0,0,v['Y']*np.exp((v['rF']-comp-v['rD'])*v['T'])*(scipy.stats.norm.cdf(d1)-1))
        
        else:
            with stats.work_precision(self.precision), np.errstate(divide='ignore', invalid='ignore'): #for T=0
                norm = np.dot(v['sigmaF'].T,v['sigmaF'])
                comp = np.dot(v['sigmaX'].T,v['sigmaF'])
                d1 = (stats.log(v['S']/ v['K'])+(v['rF']-comp+1/2*norm)*v['T'])/(stats.sqrt(norm*v['T']))
                return np.where(v['T']==0,0,v['Y']*stats.exp((v['rF']-comp-v['rD'])*v['T'])*(stats.norm_cdf(d1)-1))
        


if __name__ == '__main__':
    S = K = 30000
    Y = 1/100
    T = 2
    rD = 0.03
    rF = 0
    sigmaX = np.array((0.1,0.02))
    sigmaF = np.array((0,0.25))
    opt = quantoPut()
    opt.price(S,K,Y,T,rD,rF,sigmaX,sigmaF)
    