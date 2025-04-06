#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misc helper functions
"""
import inspect
import numpy as np

def filterKwargs(func, kwargs):
    # Get function's signature parameters
    sig = inspect.signature(func)
    return {k: kwargs[k] for k in kwargs if k in sig.parameters}

def select_rows(arr, amount):
    num_rows = arr.shape[0]
    indices = np.linspace(0, num_rows - 1, amount+1, dtype=int)  # 31 points including first and last
    return arr[indices]


def longest_list_length(data):
    return max((np.array(v).size for v in data.values() if isinstance(v, (list, tuple,np.ndarray ))), default=1)

def longest_list_shape(data):
    longest_obj = max(
        (v for v in data.values() if isinstance(v, (list, tuple, np.ndarray))), 
        key=lambda x: np.array(x).size, 
        default=None
    )
    
    if longest_obj is None or longest_obj == ():
        return (1)
    
    return longest_obj.shape if isinstance(longest_obj, np.ndarray) else (len(longest_obj),)

def split_dict(v):
    """
    For each key in v:
      - If the value is a numpy array with ndim==2, take its i-th row.
      - Otherwise, leave it as is.
    Returns a list of dicts, one per row of the 2d arrays.
    """
    # Determine the number of rows from the first encountered 2D array.
    n = 1
    for value in v.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            n = value.shape[0]
            break

    result = []
    for i in range(n):
        new_dict = {}
        for key, value in v.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                # Slice out the i-th row
                new_dict[key] = value[i, :]
            else:
                # Copy scalars and 1D arrays as-is.
                new_dict[key] = value
        result.append(new_dict)
    return result

def broadcast_dict(params, maxShape,holdScalars=True):
    """
    Broadcasts each value in the 'params' dictionary to the given maxShape.
    If a parameter is a scalar, it is expanded to an array of maxShape.
    For 1D arrays, if maxShape is 2D, it is tiled appropriately
    (assuming the 1D length matches one of the dimensions).
    Otherwise, standard broadcasting is used.
    """
    broadcasted = {}
    for key, value in params.items():
        arr = np.array(value)
        if arr.ndim == 0:
            if not holdScalars:
                # Scalar: create an array of the full shape
                arr = np.full(maxShape, arr)
            else:
                arr = value
        elif arr.ndim == 1:
            if isinstance(maxShape, tuple) and len(maxShape) == 2:
                if arr.shape[0] == maxShape[0]:
                    # Treat as a column vector and tile horizontally
                    arr = np.tile(arr.reshape(-1, 1), (1, maxShape[1]))
                elif arr.shape[0] == maxShape[1]:
                    # Treat as a row vector and tile vertically
                    arr = np.tile(arr.reshape(1, -1), (maxShape[0], 1))
                else:
                    # Fallback to broadcasting
                    arr = np.broadcast_to(arr, maxShape)
            else:
                # For 1D maxShape, simply broadcast
                arr = np.broadcast_to(np.array(arr), (maxShape))
        else:
            # For arrays with ndim>=2, broadcast if necessary
            if arr.shape != maxShape:
                arr = np.broadcast_to(arr, maxShape)
        broadcasted[key] = arr
    return broadcasted


def flatten_dict(dicts):
    flattened = {}
    for key, value in dicts.items():
        arr = np.array(value)
        if arr.ndim == 0:
            arr = value
        else:
            arr = arr.flatten()
        flattened[key] = arr
    return flattened