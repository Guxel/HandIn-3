#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default Black-Scholes Call option
"""
from .base import claim
import numpy as np
import scipy
from Helpers import stats

class call(claim):
    
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
        return np.maximum(v['S']-v['K'],0)
    
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
        if self.precision == 16:
            with np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
                d1 = (np.log(v['S'] / v['K']) + (v['r'] + 0.5 * v['sigma'] ** 2) * v['T']) / (v['sigma'] * np.sqrt(v['T']))
                d2 = d1 - v['sigma'] * np.sqrt(v['T'])
               
                price = v['S'] * scipy.stats.norm.cdf(d1) - v['K'] * np.exp(-v['r'] * v['T']) * scipy.stats.norm.cdf(d2)
                return  np.where(v['T']==0,self.payoff(**v),price)
        else:
            with stats.work_precision(self.precision), np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
                d1 = (stats.log(v['S'] / v['K']) + (v['r'] + 0.5 * v['sigma'] ** 2) * v['T']) / (v['sigma'] * stats.sqrt(v['T']))
                d2 = d1 - v['sigma'] * stats.sqrt(v['T'])
               
                price = v['S'] * stats.norm_cdf(d1) - v['K'] * stats.exp(-v['r'] * v['T']) * stats.norm_cdf(d2)
                return  np.where(v['T']==0,self.payoff(**v),price)

    def vega(self,**v):
        with stats.work_precision(self.precision),np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
            d1 = (stats.log(v['S'] / v['K']) + (v['r'] + 0.5 * v['sigma'] ** 2) * v['T']) / (v['sigma'] * stats.sqrt(v['T']))
            return np.where(v['T']==0,0,v['S'] * stats.norm_pdf(d1) * stats.sqrt(v['T']))

    def invertVol(self,**v):
        """
        Required
        ----------
        price : Price of the option
        S : Underlying Price
        K : Strike Price
        r : Constant Interest Rate
        T : Time till Maturity (NOT maturity date)
        
        Optional
        -------
        sigma : Initial Guess, default is 0.2
        max iterations : max iterations, default is 1000000
        convergence : convergence limit for maximal price diff, default is 1E-11
        
        Returns
        -------
        volatility that yields value of option
        """
        with stats.work_precision(self.precision):
            if "convergence" in v:
                convergence = v['convergence']
            else:
                convergence = 1E-5
                
            if "sigma" in v: #step 0
                sigma = v['sigma']
            else:
                sigma = np.sqrt(2*np.pi/v['T'])*(v['price']-(v['S']-v['K'])/2)/(v['S']-(v['S']-v['K'])/2)
                
            if "maxIterations" in v:
                maxIterations = v['maxIterations']
            else:
                maxIterations = 1E6
            
            if "method" in v: #simple or optimal
                method = v['method']
            else:
                method ='bach' #set to optimal
                
            if method == "optimal":
                goalLow = 2*(v['price']-v['S']+v['K']*np.exp(-v['r']*v['T']))
                goalHigh = v['price']
                goal = np.where(v['K']<v['S'],goalLow,goalHigh)
            elif method == "bach":
                v['r'] = 0
                d = (v['S']-v['K'])/(v['sigma']*stats.sqrt(v['T']))
                goalLow = -(v['S']-v['K'])*stats.erfc(d/stats.sqrt(2))+2*v['sigma']*stats.sqrt(v['T'])*stats.norm_pdf(d)
                goalHigh = (v['S']-v['K'])*stats.erfc(-d/stats.sqrt(2))+2*v['sigma']*stats.sqrt(v['T'])*stats.norm_pdf(d)
                goal = np.where(v['K']<v['S'],goalLow,goalHigh)
            else:
                goal = v['price']
                
            lb = stats.mpfy(0)
            ub = stats.mpfy(np.where(sigma * 1000>10000,sigma * 1000,10000))
            lastDiff = 0
            
            if v['T'] <= 0:
                raise Exception(f"T={v['T']} not valid, make it positive")
                
            for i in range(0,int(maxIterations)):
                if method == "bach" or method == "optimal":
                    d1 = (stats.log(v['S'] / v['K']) + (v['r'] + 0.5 * sigma ** 2) * v['T']) / (sigma * stats.sqrt(v['T']))
                    d2 = d1 - sigma * stats.sqrt(v['T'])
                    hitLow = - v['S']*stats.erfc(d1/stats.sqrt(2)) + v['K']*np.exp(-v['T']*v['r'])*stats.erfc(d2/stats.sqrt(2))
                    hitHigh = v['S']*stats.erfc(-d1/stats.sqrt(2)) - v['K']*np.exp(-v['T']*v['r'])*stats.erfc(-d2/stats.sqrt(2))
                    diff = 0.5*(goal - np.where(v['K']<v['S'],hitLow,hitHigh)) #scaled to remove 2x factor
                else:
                    pi = self.price(sigma=sigma,S=v['S'],K=v['K'],r=v['r'],T=v['T']) #step 1
                    diff =goal- pi
                    
                lb = np.where(diff >= 0, sigma, lb) #step 2
                ub = np.where(diff <= 0, sigma, ub)
                
                if np.max(ub-lb) < convergence:
                    return sigma
                
                print(f"Ite {i}, {round(np.max(ub-lb),10)}")
                if np.all(diff == lastDiff):
                    print("WARNING, Algorithm is stuck, Convergence not reached")
                    return sigma
                
                lastDiff = diff
                
                
                with np.errstate(divide='ignore', invalid='ignore',over = 'ignore'): #step 3
                    newton = sigma + diff/self.vega(sigma=sigma,S=v['S'],K=v['K'],r=v['r'],T=v['T'])
                    
                sigma = np.where((lb*1.1 < newton) & (newton < 0.9*ub),newton,(ub+lb)/2) #step 4
            print("WARNING, max itterations breached, Convergence not reached")
            return sigma
        
    def invertS(self, **v):
        """
        Required
        ----------
        price : Price of the option
        K : Strike Price
        r : Constant Interest Rate
        sigma : Constant Volatility
        T : Time till Maturity (NOT maturity date)
        
        Returns
        -------
        Price of underlying that yields value of option
        """
        SLow = np.full_like(v['price'],    0.0)  # Initialize lower bound
        SHigh = np.full_like(v['price'],v['K'])  # Initialize upper bound
        
        if "convergence" in v:
            convergence = v['convergence']
        else:
            convergence = 1E-11
        
        opt_high = self.price(S=SHigh, K=v['K'], sigma=v['sigma'], T=v['T'], r=v['r'])
        needs_adjustment = opt_high < v['price']
            
        while np.any(needs_adjustment): #fix those above upper bound
            old_SHigh = SHigh[needs_adjustment]
            SHigh[needs_adjustment] *= 10.0
            SLow[needs_adjustment] = old_SHigh  
            
            opt_high = self.price(S=SHigh, K=v['K'], sigma=v['sigma'], T=v['T'], r=v['r'])
            needs_adjustment = opt_high < v['price']
        
        diff = convergence + 1
        while diff > convergence:
            SGuess = (SLow + SHigh)/2
            optGuess = self.price(S=SGuess,K=v['K'],sigma=v['sigma'],T=v['T'],r=v['r'])
        
            SHigh = np.where(optGuess>=v['price'],SGuess,SHigh)
            SLow = np.where(optGuess<=v['price'],SGuess,SLow)
            diff = np.max(np.abs(v['price']-optGuess)/v['price'])
        return SGuess
    
    def d1(self, **v):
        with stats.work_precision(self.precision), np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
            return (stats.log(v['S'] / v['K']) + (v['r'] + 0.5 * v['sigma'] ** 2) * v['T']) / (v['sigma'] * stats.sqrt(v['T']))
    
    def d2(self, **v):
        with stats.work_precision(self.precision), np.errstate(divide='ignore', invalid='ignore'): #divide by 0 mute
            return self.d1(**v) - v['sigma'] * stats.sqrt(v['T'])

if __name__ == '__main__':
    opt = call()
    opt.price(S=100,K=5,r=0.05,sigma=0.2,T=1)
    opt.invertS(price = 95.24385288,K=5,r=0.05,sigma=0.2,T=1)
    