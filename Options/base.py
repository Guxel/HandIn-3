#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base claim
"""
from abc import ABC, abstractmethod

class claim(ABC):
    def __init__(self,precision = 16):
        self.precision = precision
    
    @abstractmethod #reqired
    def payoff(self, **v):
        pass
    
    def price(self, **v):
        print("Method not implemented")
    
    def delta(self, **v):
        print("Method not implemented")
    
    def gamma(self, **v):
        print("Method not implemented")

    def theta(self, **v):
        print("Method not implemented")
    
    def rho(self, **v):
        print("Method not implemented")
        
    def invertS(self, **v):
        print("Method not implemented")
        
    def invertsigma(self, **v):
        print("Method not implemented")