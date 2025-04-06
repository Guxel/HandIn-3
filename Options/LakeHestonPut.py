#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Put option heston Lake

"""
from .base import claim
import numpy as np
import scipy.special as sp
from .StolenAlpha import calc_alpha2 #https://github.com/tcpedersen/anderson-lake-python/blob/master/pricers.py
from Helpers import Misc 

class lakeHestonPut(claim):
    
    def payoff(self,**v):
        return np.maximum(v['K']-v['S'],0)

    def _beta(self,u,**v):
        return v['kappa']-1j*v['epsilon']*v['rho']*u
    
    def _D(self,u,beta,**v):
        return np.sqrt(beta**2 + (v['epsilon']**2)*u*(u+1j))

    def characteristic(self,u,**v):
        beta = self._beta(u,**v)
        D = self._D(u,beta,**v)
        
        r = np.where((np.real(beta)*np.real(D)+np.imag(beta)*np.imag(D))>0,
                     -v['epsilon'] ** 2 * u * (u+1j)/(beta+D),
                     beta-D)
        
        y = np.where(D!=0,
                     np.expm1(-D*v['T'])/(2*D),
                     -v['T']/2)
        
        A = v['kappa']*v['theta']/(v['epsilon']**2)*(r*v['T']-2*np.log1p(-r*y))
        B = u*(u+1j)*y/(1-r*y)
        
        return np.exp(A+B*v['V'])
    
    def Int(self,alpha,N=500,batchTol = None,**v):            
        maxShape = Misc.longest_list_shape(v)
        ns = np.arange(-N, N+1)
        
        h = np.real(sp.lambertw(2*np.pi*N)/N)
        qs = np.exp(-np.pi*np.sinh(ns*h))
        ys = 2 * qs / (1 +qs )
        
        ws = ys / (1+qs) * np.pi * np.cosh(ns*h)
        xs = 1 - ys

        keep = ~(np.isnan(xs) | (xs == 1))
        ws = ws[keep]
        xs = xs[keep]
        
        forward = np.exp(v['T']*v['r'])*v['S']
        omega = np.log(forward/v['K'])
        phi = np.pi / 12 * np.sign(omega)
        
        phi = np.where(v['rho'] - v['epsilon'] * omega /( v['V'] + v['kappa'] * v['theta'] * v['T'] )*omega< 0,phi,0)
        
        u = lambda f,x: (2/np.power(1-x,2))*f((1+x)/(1-x))
        Q = lambda z: self.characteristic(z-1j,**v)/(z*(z-1j))
        hfunc = lambda x: -1j*alpha + x*(1+1j*np.tan(phi))
        inner = lambda x:  np.real(
                            np.exp(-x*np.tan(phi)*omega)*np.exp(1j*x*omega)*Q(hfunc(x))*(1+1j*np.tan(phi))
                            )
        
        if batchTol is None:
            ws = ws[:,np.newaxis,np.newaxis]
            xs = xs[:,np.newaxis,np.newaxis]
            I = h * np.sum(np.nan_to_num(ws * u(inner,xs),0),axis=0)
        else:
            I = np.zeros(maxShape)
            N = len(xs)
            for n in range(int(N/2),N+1):  
                x = xs[n]  
                w = ws[n]
                val = np.nan_to_num(w * u(inner, x),0)
                I += val
                if np.max(np.abs(val)) < batchTol:
                    break
            for n in range(int(N/2)-1,-1,-1):  
                x = xs[n]  
                w = ws[n]
                val = np.nan_to_num(w * u(inner, x),0)
                I += val
                if np.max(np.abs(val)) < batchTol:
                    break
            I *= h
            
        return np.exp(alpha*omega) * I
    
    def Int2(self,alpha,N=500,batchTol=None,**v):
        maxShape = Misc.longest_list_shape(v)
        ns = np.arange(-N, N+1)
        
        h = np.real(sp.lambertw(2*np.pi*N)/N)
        qs = np.exp(-np.pi*np.sinh(ns*h))
        ys = 2 * qs / (1 +qs )
        
        ws = ys / (1+qs) * np.pi * np.cosh(ns*h)
        xs = 1 - ys
        
        keep = ~(np.isnan(xs) | (xs == 1))
        ws = ws[keep]
        xs = xs[keep]
        
        forward = np.exp(v['T']*v['r'])*v['S']
        omega = np.log(forward/v['K'])
        phi = np.pi / 12 * np.sign(omega)
        
        phi = np.where(v['rho'] - v['epsilon'] * omega /( v['V'] + v['kappa'] * v['theta'] * v['T'] )*omega< 0,phi,0)
        
        u = lambda f,x: (2/np.power(1-x,2))*f((1+x)/(1-x))
        Q = lambda z: self.characteristic(z-1j,**v)/(z*(z-1j))
        hfunc = lambda x: -1j*alpha + x*(1+1j*np.tan(phi))
        inner = lambda x:  np.real(
                            (alpha+x*(1j-np.tan(phi)))*np.exp(-x*np.tan(phi)*omega)*np.exp(1j*x*omega)*Q(hfunc(x))*(1+1j*np.tan(phi))
                            )

        if batchTol is None:
            ws = ws[:,np.newaxis,np.newaxis]
            xs = xs[:,np.newaxis,np.newaxis]
            I2 = h * np.sum(np.nan_to_num(ws * u(inner,xs),0),axis=0)
        else:
            I2 = np.zeros(maxShape)
            N = len(xs)
            for n in range(int(N/2),N+1):  
                x = xs[n]  
                w = ws[n]
                val = np.nan_to_num(w * u(inner, x),0)
                I2 += val
                if np.max(np.abs(val)) < batchTol:
                    break
            for n in range(int(N/2)-1,-1,-1):  
                x = xs[n]  
                w = ws[n]
                val = np.nan_to_num(w * u(inner, x),0)
                I2 += val
                if np.max(np.abs(val)) < batchTol:
                    break
            I2 *= h
            
        return np.exp(alpha*omega) * I2
    
    def _R(self,forward,alpha,**v):
        return np.where(alpha<=0,forward,0)-np.where(alpha<=-1,v['K'],0)-(np.where(alpha==0,forward,0)-np.where(alpha==-1,v['K'],0))/2

    def _RdS(self,forward,alpha,**v):
        return np.where(alpha<=0,1,0)-np.where(alpha==0,1/2,0)
    
    def price(self,**v):
        maxShape = Misc.longest_list_shape(v)
        v = Misc.broadcast_dict(v,maxShape)
        if not 'lazyAlpha' in v:
            v['lazyAlpha'] = False
        with np.errstate(divide='ignore', invalid='ignore'):
            forward = np.exp(v['T']*v['r'])*v['S']
            alpha = calc_alpha2(v['lazyAlpha'],**v)
            R = self._R(forward,alpha,**v)
            call = np.exp(-v['r']*v['T'])*(R-forward/np.pi * self.Int(alpha,**v))
            return  np.where(v['T']==0,self.payoff(**v),v['K']*np.exp(-v['r'] * v['T'])-v['S']+call)

    def delta(self,i=0,**v):
        maxShape = Misc.longest_list_shape(v)
        v = Misc.broadcast_dict(v,maxShape)
        if not 'lazyAlpha' in v:
            v['lazyAlpha'] = False
        with np.errstate(divide='ignore', invalid='ignore'):
            forward = np.exp(v['T']*v['r'])*v['S']
            alpha = calc_alpha2(v['lazyAlpha'],**v)
            Reffect = self._RdS(forward,alpha,**v)
            IntEffect = - (self.Int(alpha, **v)+self.Int2(alpha, **v))/np.pi
            return  np.real(np.where(v['T']==0,0,-1+Reffect+IntEffect))


if __name__ == '__main__':
    putOptLake = lakeHestonPut()
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

    r=0.02
    putOptLake.characteristic(10,S=A0,K=K,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
    price = putOptLake.price(N=250,S=A0,K=K,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
    print(price)
    putOptLake.delta(N=250,S=A0,K=K,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)