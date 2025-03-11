#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misc helper functions
"""
import inspect

def filterKwargs(func, kwargs):
    # Get function's signature parameters
    sig = inspect.signature(func)
    return {k: kwargs[k] for k in kwargs if k in sig.parameters}