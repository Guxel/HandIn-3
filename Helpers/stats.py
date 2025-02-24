
"""
Uses statistical methods with increased precision!
"""
import mpmath as mp
import numpy as np

def _norm_cdf(x):
    """
    High-precision computation of the standard normal CDF using mpmath.
    """
    return 0.5 * (1 + mp.erf(x / mp.sqrt(2)))

def _norm_pdf(x):
    """
    High-precision computation of the standard normal PDF using mpmath.
    """
    return mp.exp(-0.5 * x**2) / mp.sqrt(2 * mp.pi)

def set_precision(precision:int):
    mp.mp.dps = precision

norm_cdf = np.vectorize(_norm_cdf)
norm_pdf = np.vectorize(_norm_pdf)
log = np.vectorize(mp.log)
sqrt = np.vectorize(mp.sqrt)
exp = np.vectorize(mp.exp)
mpfy = np.vectorize(lambda x: mp.mpf(float(x)))
work_precision = mp.workdps