#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Put option heston naive
"""
from .base import claim
import numpy as np
from Helpers import Misc 

class naiveHestonPut(claim):
    def __init__(self,precision = 16,fixD=True):
        self.precision = precision
        self.fixD = fixD
    
    def payoff(self,**v):
        return np.maximum(v['K']-v['S'],0)
    
    def _us(self):
        return np.array([1/2,-1/2])

    def _bs(self,**v):
        return np.array([v['kappa'] + v['lambd'] - v['rho'] * v['epsilon'],v['kappa'] + v['lambd']])
    
    def _ds(self,phi,bs,us,**v):
        return np.where(self.fixD,-1,1) * np.sqrt(np.power(v['rho'] * v['epsilon']*phi*1j-bs,2) - (v['epsilon']**2) *(2*us*phi*1j-(phi**2)) )
    
    def _gs(self,phi,ds,bs,**v):
        return (bs - v['rho'] * v['epsilon']*phi*1j + ds)/(bs- v['rho'] * v['epsilon']*phi*1j - ds)

    def _Ds(self,phi,gs,ds,bs,tau,**v):
        expon = np.exp(ds*tau)
        return ((bs - v['rho'] * v['epsilon']*phi*1j + ds)/(v['epsilon']**2)) * ((1-expon)/(1-gs*expon))
    
    def _Cs(self,phi,gs,ds,bs,tau,**v):
        expon = np.exp(ds*tau)
        return v['r']*phi*1j*tau+((v['kappa']*v['theta'])/(v['epsilon']**2))*((bs-v['rho']*v['epsilon']*phi*1j+ds)*tau-2*np.log((1-gs*expon)/(1-gs)))
    
    def characteristic(self,phi,bs=None,**v):
        if bs is None:
            x,y = phi.shape
            bs =  np.tile(self._bs(**v)[None, :], (x, 1))
        ds = self._ds(phi, bs, self._us(), **v)
        gs = self._gs(phi, ds, bs, **v)
        with np.errstate(over='ignore', invalid='ignore'):
            return np.exp(
            self._Cs(phi, gs, ds, bs, v['T'], **v) +
            self._Ds(phi, gs, ds, bs, v['T'], **v) * v['V'] +
            1j * phi * np.log(v['S'])  
        )
    
    def Ps(self,phiLim,n,**v):
        maxLen = Misc.longest_list_length(v)
        
        v['K'] = np.asarray(v['K'])
        if v['K'].ndim == 0:  
            v['K'] = np.array([v['K']])
            
        phis = np.tile(np.linspace(0, phiLim,n+2)[1:-1,None, None], (1,2,maxLen)).transpose(2,0,1)
        logK = np.tile(np.log(v['K'])[:,None,None],(1,2,n)).transpose(0,2,1)
        bs = np.tile(self._bs(**v)[None, :, None], (n, 1, maxLen)).transpose(2, 0, 1)
        return 1/2 + phiLim/(np.pi*n) * np.sum(np.nan_to_num(np.real(
                (np.exp(-1j*phis*logK)*self.characteristic(phis,bs,**v))
                /(1j*phis)
                )),axis=1)
    
    def price(self,**v):
        if 'nPartitions' in v:
            v['n'] = v['nPartitions']
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Ps = self.Ps(**v)
            return  np.where(v['T']==0,self.payoff(**v),v['K']*np.exp(-v['r']*v['T'])*(1-Ps[:,1])-v['S']*(1-Ps[:,0]))

if __name__ == '__main__':
    putOpt = naiveHestonPut()
    
    r = 0.02
    std = 0.2
    A0 = S0 = 1
    T = 30
    K = np.exp(r*T)
    a = 0.5
    
    V0 = theta = 0.2 ** 2
    kappa = 2
    epsilon = 1
    rho = -0.5
    lambd=0
    
    phiLim = 15
    n = 50000
    putOpt = naiveHestonPut(fixD=True) 
    r=0.02
    
    us = putOpt._us()
    bs = putOpt._bs(phiLim=phiLim,n=n,  S=A0,K=K,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
    
    putOpt._ds(10,bs,us,phiLim=phiLim,n=n,  S=A0,K=K,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
    putOpt.price(phiLim=phiLim,n=n,  S=A0,K=K,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
