# -*- coding: utf-8 -*-
import numpy as np

from scipy.optimize import brentq, fminbound

def calc_alpha2(lazy = False,**v) -> np.ndarray:
    # a if K is an array
    if lazy:
        return -1
    print("Getting alpha")
    alpha_mins, alpha_maxs = alpha_min_max2(lazy,**v)
    
    print("Min-Max Alpha Found")
    #if lazy:
    #    return np.maximum(-1.5,alpha_mins)

    forwards = np.exp(v['T'] * v['r']) * v['S']
    
    eps = np.sqrt(np.finfo(np.float64).eps)
    omegeas = np.log(forwards / v['K'])

    alpha, val = locate_optimal_alpha2(alpha_mins, -1 - eps,omegeas,lazy,**v)    
    alpha2, val2 = locate_optimal_alpha2(eps, alpha_maxs,omegeas,lazy,**v)
    
    alphaRet = alpha
    alphaRet = np.where(omegeas<0,alpha2,alphaRet)
    alphaRet = np.where(((omegeas<0)&(val2>9))&((np.abs(alpha_maxs) <= 1e-3)|(v['kappa'] - v['rho'] * v['epsilon'] > 0))
                        ,alpha,alphaRet)
    alphaRet = np.where(v['T']==0,np.maximum(-1.5,alpha_mins),alphaRet)
    
    print("alphas found")
    return np.nan_to_num(alphaRet,nan=alpha_mins)

def log_cf_real2(alpha, tau,rho,kappa,epsilon,theta,V) -> float:
        # Evaluation of ln HestomModel.cf(-1j * (1 + alpha))
        beta = kappa - rho * epsilon * (1 + alpha)
        Dsq = beta**2 - epsilon**2 * (1 + alpha) * alpha
        
        D = np.sqrt(Dsq)
        coshdt = np.cosh(D * tau / 2)
        sinhdt = np.sinh(D * tau / 2) / D
        nume = coshdt + beta * sinhdt

        x = np.sqrt(-Dsq)
        coshdt2 = np.cos(x * tau / 2)
        sinhdt2 = np.sin(x * tau / 2) / x
        nume = np.where(Dsq > 0,nume, coshdt2 + beta * sinhdt2)
        

        A = kappa * theta / epsilon**2 *\
            (beta * tau - np.log(nume**2))
        B = alpha * (1 + alpha) * sinhdt / nume

        return A + B * V
    
def make_obj_func(T, rho, kappa, epsilon, theta, V, omega):
    return lambda alpha: (
        log_cf_real2(alpha, T, rho, kappa, epsilon, theta, V)
        - np.log(alpha * (alpha + 1))
        + alpha * omega
    )

def optimize_alpha(T, rho, kappa, epsilon, theta, V, omega, a, b):
    if T == 0 or V == 0:
        return 0, 0
    
    obj_func = make_obj_func(T, rho, kappa, epsilon, theta, V, omega)
    alpha_opt, val = fminbound(obj_func, a, b, full_output=True)[0:2]
    return alpha_opt, val

def optimize_alphaLazy(T, rho, kappa, epsilon, theta, V, omega, a, b):
    if T == 0 or V == 0:
        return 0, 0
    
    obj_func = make_obj_func(T, rho, kappa, epsilon, theta, V, omega)
    alpha_opt, val = fminbound(obj_func, a, b,xtol=1e-3, full_output=True)[0:2]
    return alpha_opt, val

def locate_optimal_alpha2(a, b,omegas,lazy=False,**v):
    if lazy:
        vec_optimize_alpha = np.vectorize(
            optimize_alphaLazy, 
            otypes=[float, float]
        )
    else:
        vec_optimize_alpha = np.vectorize(
            optimize_alpha, 
            otypes=[float, float]
        )
    alphas, vals = vec_optimize_alpha(
        v['T'], v['rho'], v['kappa'], v['epsilon'], v['theta'], v['V'], omegas,
        a=a, b=b
    )
    return np.real(alphas), np.real(vals)

def k_plus_minus2(x: float, sign: int, **v) -> float:
    A = v['epsilon'] - 2 * v['rho'] * v['kappa']
    B = (v['epsilon'] - 2 * v['rho'] * v['kappa'])**2 +\
        4 * (v['kappa']**2 + x**2 / v['T']**2) * (1 - v['rho']**2)
    C = 2 * v['epsilon'] * (1 - v['rho']**2)

    return (A + sign * np.sqrt(B)) / C


def critical_moments_func2(k: float, kminus,kplus,tau,rho,kappa,epsilon) -> float:  
    beta = kappa - rho * epsilon * k
    D = np.sqrt(beta**2 + epsilon**2 * (-1j * k) * ((-1j * k) + 1j))

    return np.where((k > kplus)|(k < kminus),np.cos(np.abs(D) * tau / 2) + \
                                                beta * np.sin(np.abs(D) * tau / 2) / np.abs(D) 
                                                ,np.cosh(D.real * tau/ 2) + \
                                                    beta * np.sinh(D.real * tau / 2) / D.real)

def solveBrent(a, b, kminus, kplus, tau,rho,kappa,epsilon):
    if tau == 0:
        return 0
    return brentq(
        critical_moments_func2,
        a,
        b,
        args=(kminus, kplus, tau,rho,kappa,epsilon)
    )

def solveBrentLazy(a, b, kminus, kplus, tau,rho,kappa,epsilon):
    if tau == 0:
        return 0
    return brentq(
        critical_moments_func2,
        a,
        b,
        args=(kminus, kplus, tau,rho,kappa,epsilon),
        rtol = 1e-4
    )

vec_brent = np.vectorize(
                solveBrent,
                otypes=[float]
            )

vec_brentLazy = np.vectorize(
                solveBrentLazy,
                otypes=[float]
            )

def alpha_min_max2(lazy,**v) -> (float, float):
    kminus = k_plus_minus2(0, -1, **v)
    kplus = k_plus_minus2(0, 1, **v)
    
    # The interval in which to locate k is technically an open interval,
    # so a small number is added/substracted to/from the boundaries.
    eps = np.sqrt(np.finfo(np.float64).eps)
    
    # Find kmin
    kmin2pi = k_plus_minus2(2 * np.pi, -1, **v)

    if lazy:
        kmin = 1e-4 + vec_brentLazy(kmin2pi + eps, kminus - eps,kminus, kplus,v['T'],v['rho'],v['kappa'],v['epsilon'])
    else:
        kmin = vec_brent(kmin2pi + eps, kminus - eps,kminus, kplus,v['T'],v['rho'],v['kappa'],v['epsilon'])
        
    # Find kmax
    kps = v['kappa'] - v['rho'] * v['epsilon']
    kplus2 = k_plus_minus2(np.pi, 1, **v)
    kplus3 = k_plus_minus2(2*np.pi, 1, **v)
    
    a = kplus
    
    b = kplus2
    b = np.where(kps > 0,kplus3,b)
    
    T = -2 / (v['kappa'] - v['rho'] * v['epsilon'] * kplus)
    a = np.where((kps < 0) & (v['T']>=T),1,a)
    b = np.where((kps < 0) & (v['T']>=T),kplus,b)
    
    if lazy:
        kmax = -1e-4 + vec_brentLazy(a + eps, b - eps,kminus, kplus,v['T'],v['rho'],v['kappa'],v['epsilon'])
    else:
        kmax = vec_brent(a + eps, b - eps,kminus, kplus,v['T'],v['rho'],v['kappa'],v['epsilon'])

    return kmin - 1, kmax - 1

