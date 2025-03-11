#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for Longstaff-Schwartz
"""
import numpy as np
from Options.base import claim
import Helpers.Misc as misc
from sklearn.linear_model import LinearRegression
from numpy.polynomial.laguerre import lagvander

def MatrixReg(X,Y,sklearn = False):
    if sklearn:
        model = LinearRegression(fit_intercept=False)
        model.fit(X, Y)
        return model.coef_
    else:
        try:
            return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),Y)
        except np.linalg.LinAlgError as e:
            print(f"linAlgError, attempting pseudo inverse: {e}")
            return np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),Y).T

def getDesign(S,regMethod :str):
    match regMethod.lower():
        case "intercept":
            return np.column_stack(np.ones(S.shape[0])).T
        case "simple":
            return np.column_stack((np.ones(S.shape[0]), S, S**2))
        case "1d":
           return np.column_stack((np.ones(S.shape[0]), S))
        case "3d":
           return np.column_stack((np.ones(S.shape[0]), S, S**2, S**3))
        case "4d":
           return np.column_stack((np.ones(S.shape[0]), S, S**2, S**3, S**4))
        case "5d":
            return np.column_stack((np.ones(S.shape[0]), S, S**2, S**3, S**4, S**5))
        case "laguerre2":
            n = 2
            vander = lagvander(S, n) 
            return np.column_stack((np.ones(S.shape[0]),vander * np.exp(-S[:, None] / 2)))
        case "laguerre3":
            n = 3
            vander = lagvander(S, n) 
            return np.column_stack((np.ones(S.shape[0]),vander * np.exp(-S[:, None] / 2)))
        case "laguerre4":
            n = 4
            vander = lagvander(S, n) 
            return np.column_stack((np.ones(S.shape[0]),vander * np.exp(-S[:, None] / 2)))
        case "laguerre5":
            n = 5
            vander = lagvander(S, n) 
            return np.column_stack((np.ones(S.shape[0]),vander * np.exp(-S[:, None] / 2)))
        case "laguerre10":
            n = 10
            vander = lagvander(S, n) 
            return np.column_stack((np.ones(S.shape[0]),vander * np.exp(-S[:, None] / 2)))
        case "laguerre50":
            n = 50
            vander = lagvander(S, n) 
            return np.column_stack((np.ones(S.shape[0]),vander * np.exp(-S[:, None] / 2)))
        case "laguerre500":
            n = 500
            vander = lagvander(S, n) 
            return np.column_stack((np.ones(S.shape[0]),vander * np.exp(-S[:, None] / 2)))
    raise ValueError(f"Invalid regMethod: {regMethod}")

def priceDefault(option: claim,Sim,dt,r,regMethod = "simple",sklearn=False,moreInfo = False,**v):
    """
    Use Longstaf-Schwartz for pricing American options
    """
    n,T = Sim.shape
    CF = np.zeros((n,T-1))
    
    payOff = option.payoff(S=Sim[:,-1],**v)
    CF[:,-1] = payOff
    DF = np.exp(-r * dt * np.arange(1,T))
    B = []
    
    for t in range(T-2,0,-1):
        payOff = option.payoff(S=Sim[:,t],**v)
        ITM = payOff>0
        
        if np.any(ITM):
            X = getDesign(Sim[ITM,t],regMethod)
            regCoef = MatrixReg(X,np.dot(CF[ITM,t:],DF[:T-t-1]),sklearn)
            
            excercise = np.repeat(False,n)
            excercise[ITM] = payOff[ITM] > np.dot(X,regCoef)
            
            CF[excercise,:] = 0
            CF[excercise,t-1] = payOff[excercise]
            
            B.append(regCoef)
        else:
            B.append([])
        
        
    price = np.mean(np.dot(CF,DF))
    if moreInfo:
        return price, CF, B
    else:
        return price
    
def priceDefaultFixed(option: claim,Sim,dt,r,B,regMethod= "simple",moreInfo = False,**v):
    """
    Use Longstaf-Schwartz for pricing American options
    """
    n,T = Sim.shape
    CF = np.zeros((n,T-1))
    
    payOff = option.payoff(S=Sim[:,-1],**v)
    CF[:,-1] = payOff
    DF = np.exp(-r * dt * np.arange(1,T))
    
    for t in range(T-2,0,-1):
        payOff = option.payoff(S=Sim[:,t],**v)
        ITM = payOff>0
        
        if np.any(ITM):
            X = getDesign(Sim[ITM,t],regMethod)
            regCoef = B[t-1]
            
            excercise = np.repeat(False,n)
            excercise[ITM] = payOff[ITM] > np.dot(X,regCoef)
            
            CF[excercise,:] = 0
            CF[excercise,t-1] = payOff[excercise]
        
        
    price = np.mean(np.dot(CF,DF))
    if moreInfo:
        return price, CF
    else:
        return price


def getFirstRegParams(option: claim,Sim,dt,r,regMethod = "simple",sklearn=False,**v):
    n,T = Sim.shape
    
    payOff = option.payoff(S=Sim[:,-1],**v)

    payOff2 = option.payoff(S=Sim[:,-2],**v)
    ITM = payOff2>0
        
    X = getDesign(Sim[ITM,-2],regMethod)
    return MatrixReg(X,payOff[ITM]*np.exp(-r * dt),sklearn)

def getFirstRegMSE(option: claim,Sim,dt,r,regMethod = "simple",sklearn=False,**v):
    n,T = Sim.shape
    
    payOff = option.payoff(S=Sim[:,-1],**v)

    payOff2 = option.payoff(S=Sim[:,-2],**v)
    ITM = payOff2>0
        
    X = getDesign(Sim[ITM,-2],regMethod)
    beta = MatrixReg(X,payOff[ITM]*np.exp(-r * dt),sklearn)
    return np.mean(np.power(payOff[ITM]*np.exp(-r * dt)-np.dot(X,beta),2))


def priceDefaultMultiple(option: claim,simMethod,dt,r,n,m,regMethod = "simple",sklearn=False,B=None,antihetic=True,**v):
    pList = []
    simV = misc.filterKwargs(simMethod,v)
    if B is None:
        for i in range(0,n):
            print(i)
            sim = simMethod(delta=dt,mu=r,m=m,**simV,antihetic = antihetic).T
            pList.append(priceDefault(option,sim,dt,r,regMethod,sklearn,**v))
    else:
        for i in range(0,n):
            print(i)
            sim = simMethod(delta=dt,mu=r,m=m,**simV,antihetic = antihetic).T
            pList.append(priceDefaultFixed(option,sim,dt,r,regMethod=regMethod,B=B,sklearn=sklearn,**v))
    return pList

