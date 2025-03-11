"""
Code script for FinKont2
Hand-In 2
Gustav Dyhr
msh606
2025-03-04
"""
#%% Global
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set_theme()
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 800    

sys.exit()

#%% 1 - log returns distribution
import Helpers.DataManager as dm
from scipy.optimize import minimize

sim = np.array([
    [1,1.09,1.08,1.34],
    [1,1.16,1.26,1.54],
    [1,1.22,1.07,1.03],
    [1,0.93,0.97,0.92],
    [1,1.11,1.56,1.52],
    [1,0.76,0.77,0.90],
    [1,0.92,0.84,1.01],
    [1,0.88,1.22,1.34]
    ])

logReturns = np.log(sim/dm.shift(sim,1,axis=1))[:,1:]

stdRV = np.std(logReturns.flatten(),ddof=1) #simple std
seRV = stdRV/np.sqrt(2*(logReturns.size-1))

def logLik(sigma,x):
    return sum(-np.log(sigma*np.sqrt(2*np.pi))-1/2*np.power((x-0.06)/sigma+0.5*sigma,2))

def minusLogLik(sigma,x):
    return -logLik(sigma,x)

MLE = minimize(minusLogLik,stdRV,args = logReturns.flatten(),bounds = [(0.001,None)])
stdML = MLE.x[0]

def computeSE(sigma, x):
    n = len(x)
    secoundDeriv = (n / sigma**2) \
        + np.sum(x * (0.36 - 3 * x) - 0.25 * sigma**4 - 0.0108)/ sigma**4
    return np.sqrt(-1 / secoundDeriv)

seML = computeSE(sigma = stdML, x = logReturns.flatten())


#%% 2
import Helpers.LongstaffSchwartz as LSM
from Helpers.SimulateProcess import Simulate_GBM
from Options.Put import put
putOpt = put()
delta = 3
T = 3
r = 0.06
stdReal = 0.15 

#%% 2 - price EU (a bit slow)
delta = 3
pricesEUReal = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=int(10000000*delta/3),n=1000, sigma = stdReal)
pricesEURV = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=int(10000000*delta/3),n=1000, sigma = stdRV)
pricesEUML = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=int(10000000*delta/3),n=1000, sigma = stdML)

bsReal = putOpt.price(S=1,r = r, K=1.1,T=T,sigma= stdReal)
bsRV = putOpt.price(S=1,r = r, K=1.1,T=T,sigma= stdRV)
bsML = putOpt.price(S=1,r = r, K=1.1,T=T,sigma= stdML)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(pricesEUReal, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axs[0].axvline(x=bsReal, color='red', linestyle='--', linewidth=2,label = "BS")
axs[0].axvline(x=np.mean(pricesEUReal), color='blue', linestyle='--', linewidth=2,label = "Mean")
axs[0].set_title("$\sigma_{0.15}$")
axs[0].set_xlabel("$\pi_{EU}$")
axs[0].set_ylabel("Frequency")
axs[0].legend(loc="upper left")
axs[0].text(0.65, 0.95, f'Mean: {np.mean(pricesEUReal):.4f}\nStd: {np.std(pricesEUReal,ddof=1):.6f}', 
            transform=axs[0].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

axs[1].hist(pricesEURV, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axs[1].axvline(x=bsRV, color='red', linestyle='--', linewidth=2)
axs[1].axvline(x=np.mean(pricesEURV), color='blue', linestyle='--', linewidth=2)
axs[1].set_title("$\sigma_{RV}$")
axs[1].set_xlabel("$\pi_{EU}$")
axs[1].text(0.65, 0.95, f'Mean: {np.mean(pricesEURV):.4f}\nStd: {np.std(pricesEURV,ddof=1):.6f}', 
            transform=axs[1].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

axs[2].hist(pricesEUML, bins=50, color='salmon', edgecolor='black', alpha=0.7)
axs[2].axvline(x=bsML, color='red', linestyle='--', linewidth=2)
axs[2].axvline(x=np.mean(pricesEUML), color='blue', linestyle='--', linewidth=2)
axs[2].set_title("$\sigma_{ML}$")
axs[2].set_xlabel("$\pi_{EU}$")
axs[2].text(0.65, 0.95, f'Mean: {np.mean(pricesEUML):.4f}\nStd: {np.std(pricesEUML,ddof=1):.6f}', 
            transform=axs[2].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

plt.show()

print(bsReal - np.median(pricesEUReal))
print(bsReal - np.mean(pricesEUReal))

print(bsRV - np.median(pricesEURV))
print(bsRV - np.mean(pricesEURV))

print(bsML - np.median(pricesEUML))
print(bsML - np.mean(pricesEUML))

#%% 2 - Price Bermudan (slow)
delta = 1
pricesBMReal = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=int(10000000*delta/3),n=1000, sigma = stdReal)
pricesBMRV = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=int(10000000*delta/3),n=1000, sigma = stdRV)
pricesBMML = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=int(10000000*delta/3),n=1000, sigma = stdML)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(pricesBMReal, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axs[0].axvline(x=np.mean(pricesBMReal), color='blue', linestyle='--', linewidth=2,label = "Mean")
axs[0].set_title("$\sigma_{0.15}$")
axs[0].set_xlabel("$\pi_{BM}$")
axs[0].set_ylabel("Frequency")
axs[0].legend(loc="upper left")
axs[0].text(0.65, 0.95, f'Mean: {np.mean(pricesBMReal):.4f}\nStd: {np.std(pricesBMReal,ddof=1):.6f}', 
            transform=axs[0].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))


axs[1].hist(pricesBMRV, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axs[1].axvline(x=np.mean(pricesBMRV), color='blue', linestyle='--', linewidth=2)
axs[1].xaxis.set_major_locator(MaxNLocator(nbins=5))   #fix ticks
axs[1].set_title("$\sigma_{RV}$")
axs[1].set_xlabel("$\pi_{BM}$")
axs[1].text(0.65, 0.95, f'Mean: {np.mean(pricesBMRV):.4f}\nStd: {np.std(pricesBMRV,ddof=1):.6f}', 
            transform=axs[1].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

axs[2].hist(pricesBMML, bins=50, color='salmon', edgecolor='black', alpha=0.7)
axs[2].axvline(x=np.mean(pricesBMML), color='blue', linestyle='--', linewidth=2)
axs[2].set_title("$\sigma_{ML}$")
axs[2].set_xlabel("$\pi_{BM}$")
axs[2].text(0.65, 0.95, f'Mean: {np.mean(pricesBMML):.4f}\nStd: {np.std(pricesBMML,ddof=1):.6f}', 
            transform=axs[2].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

plt.show()

print(np.mean(pricesBMReal))
print(np.mean(pricesBMRV))
print(np.mean(pricesBMML))

#%% 2 - Price American (very slow)
delta = 1/252
m = 10000
pricesAMReal = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=1000, sigma = stdReal)
pricesAMRV = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=1000, sigma = stdRV)
pricesAMML = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=1000, sigma = stdML)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(pricesAMReal, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axs[0].axvline(x=np.mean(pricesAMReal), color='blue', linestyle='--', linewidth=2,label = "Mean")
axs[0].set_title("$\sigma_{0.15}$")
axs[0].set_xlabel("$\pi_{AM}$")
axs[0].set_ylabel("Frequency")
axs[0].legend(loc="upper left")
axs[0].text(0.65, 0.95, f'Mean: {np.mean(pricesAMReal):.4f}\nStd: {np.std(pricesAMReal,ddof=1):.6f}', 
            transform=axs[0].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

axs[1].hist(pricesAMRV, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axs[1].axvline(x=np.mean(pricesAMRV), color='blue', linestyle='--', linewidth=2)
axs[1].set_title("$\sigma_{RV}$")
axs[1].set_xlabel("$\pi_{AM}$")
axs[1].text(0.65, 0.95, f'Mean: {np.mean(pricesAMRV):.4f}\nStd: {np.std(pricesAMRV,ddof=1):.6f}', 
            transform=axs[1].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

axs[2].hist(pricesAMML, bins=50, color='salmon', edgecolor='black', alpha=0.7)
axs[2].axvline(x=np.mean(pricesAMML), color='blue', linestyle='--', linewidth=2)
axs[2].set_title("$\sigma_{ML}$")
axs[2].set_xlabel("$\pi_{AM}$")
axs[2].text(0.65, 0.95, f'Mean: {np.mean(pricesAMML):.4f}\nStd: {np.std(pricesAMML,ddof=1):.6f}', 
            transform=axs[2].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

plt.show()

print(np.mean(pricesAMReal))
print(np.mean(pricesAMRV))
print(np.mean(pricesAMML))

#%% 2 - Compare Regressions
from Helpers.SimulateProcess import Simulate_GBM
delta = 1/252
simulation = Simulate_GBM(mu=r,delta=delta,T=3,S0=1,m=100000,sigma=stdReal, antihetic = True).T
p, CF, B = LSM.priceDefault(putOpt, simulation, dt = delta, r = r
                                  , K=1.1,regMethod="Laguerre3",sklearn=True, moreInfo = True)

#CF = np.loadtxt("temp.csv")
#simulation = np.loadtxt("temp2.csv")
DF = np.exp(-r * delta * np.arange(1,766 ) )

PayOff = putOpt.payoff(S=simulation[:,-2],K=1.1)
PayOffN = putOpt.payoff(S=simulation[:,-1],K=1.1)

ITM = PayOff > 0
y = np.dot(PayOffN[ITM],DF[-1])

X2 = LSM.getDesign(simulation[ITM,-2],'simple')
beta2 = LSM.MatrixReg(X2,y,True)
y2f = np.dot(X2,beta2)

X4 = LSM.getDesign(simulation[ITM,-2],'Laguerre3')
beta4 = LSM.MatrixReg(X4,y,True)
y4 = np.dot(X4,beta4)

X5 = LSM.getDesign(simulation[ITM,-2],'Laguerre5')
beta5 = LSM.MatrixReg(X5,y,True)
y5 = np.dot(X5,beta5)

PayOff2 = putOpt.payoff(S=simulation[:,1],K=1.1)

ITM2 = PayOff2 > 0
y2 = np.dot(CF[ITM,1:],DF[1:])

X22 = LSM.getDesign(simulation[ITM,1],'simple')
beta22 = LSM.MatrixReg(X22,y2,True)
y22 = np.dot(X22,beta22)

X42 = LSM.getDesign(simulation[ITM,1],'Laguerre3')
beta42 = LSM.MatrixReg(X42,y2,True)
y42 = np.dot(X42,beta42)

X52 = LSM.getDesign(simulation[ITM,1],'Laguerre5')
beta52 = LSM.MatrixReg(X52,y2,True)
y52 = np.dot(X52,beta52)


fig, ax1 = plt.subplots(1,2)


ax1[0].plot(simulation[ITM, -2], y, 'bo', label="Disc Payoff", markersize=1)
ax1[0].plot(simulation[ITM, -2], y2f, 'yo', label="2 poly", markersize=1)
ax1[0].plot(simulation[ITM, -2], y4, 'o',color='black', label="3 Laguerre poly", markersize=1)
ax1[0].plot(simulation[ITM, -2], y5, 'ro', label="4 Laguerre poly", markersize=1)
ax1[0].set_ylabel("$")

x_vec = np.linspace(simulation[ITM,-2].min(), simulation[ITM,-2].max(), 200)
ax1[0].plot(x_vec, putOpt.payoff(S=x_vec,K=1.1), 'k-', linewidth=1, label="Payoff Now")

ax1[0].legend(loc="lower left")
ax1[0].set_xlabel("X")
ax1[0].set_title("Fits for t=T-1")

ax1[1].plot(simulation[ITM, 1], y2, 'bo', label="Disc Payoff", markersize=1)
ax1[1].plot(simulation[ITM, 1], y22, 'yo', label="2 poly", markersize=1)
ax1[1].plot(simulation[ITM, 1], y42, 'o',color='black', label="3 Laguerre poly", markersize=1)
ax1[1].plot(simulation[ITM, 1], y52, 'ro', label="5 Laguerre poly", markersize=1)

x_vec = np.linspace(simulation[ITM,1].min(), simulation[ITM,1].max(), 200)
ax1[1].plot(x_vec, putOpt.payoff(S=x_vec,K=1.1), 'k-', linewidth=1, label="Payoff Now")


ax1[1].set_xlabel("X")
ax1[1].set_title("Fits for t=1")

plt.show()

#%% 2 - Price American BETTER (very slow)
delta = 1/252
m = 100000
pricesAMReal = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=100, sigma = stdReal,regMethod="Laguerre3",sklearn=True)
pricesAMRV = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=100, sigma = stdRV,regMethod="Laguerre3",sklearn=True)
pricesAMML = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=100, sigma = stdML,regMethod="Laguerre3",sklearn=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].hist(pricesAMReal, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
axs[0].axvline(x=np.mean(pricesAMReal), color='blue', linestyle='--', linewidth=2,label = "Mean")
axs[0].set_title("$\sigma_{0.15}$")
axs[0].set_xlabel("$\pi_{AM}$")
axs[0].set_ylabel("Frequency")
axs[0].legend(loc="upper left")
axs[0].text(0.65, 0.95, f'Mean: {np.mean(pricesAMReal):.4f}\nStd: {np.std(pricesAMReal,ddof=1):.6f}', 
            transform=axs[0].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

axs[1].hist(pricesAMRV, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
axs[1].axvline(x=np.mean(pricesAMRV), color='blue', linestyle='--', linewidth=2)
axs[1].set_title("$\sigma_{RV}$")
axs[1].set_xlabel("$\pi_{AM}$")
axs[1].text(0.65, 0.95, f'Mean: {np.mean(pricesAMRV):.4f}\nStd: {np.std(pricesAMRV,ddof=1):.6f}', 
            transform=axs[1].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

axs[2].hist(pricesAMML, bins=10, color='salmon', edgecolor='black', alpha=0.7)
axs[2].axvline(x=np.mean(pricesAMML), color='blue', linestyle='--', linewidth=2)
axs[2].set_title("$\sigma_{ML}$")
axs[2].set_xlabel("$\pi_{AM}$")
axs[2].text(0.65, 0.95, f'Mean: {np.mean(pricesAMML):.4f}\nStd: {np.std(pricesAMML,ddof=1):.6f}', 
            transform=axs[2].transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

plt.show()

print(np.mean(pricesAMReal))
print(np.mean(pricesAMRV))
print(np.mean(pricesAMML))

#%% 3
import Helpers.LongstaffSchwartz as LSM
from Options.Put import put
putOpt = put()
delta = 1
r = 0.06
T = 3
sim = np.array([
    [1,1.09,1.08,1.34],
    [1,1.16,1.26,1.54],
    [1,1.22,1.07,1.03],
    [1,0.93,0.97,0.92],
    [1,1.11,1.56,1.52],
    [1,0.76,0.77,0.90],
    [1,0.92,0.84,1.01],
    [1,0.88,1.22,1.34]
    ])
pricesBMReal = 0.0926  


p0D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "intercept") 
p1D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "1d") 
pSimple = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1) 
p3D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "3d") 
p4D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "4d") 
p5D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "5d") 

r0D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "intercept") 
r1D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "1d") 
rSimple = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1) 
r3D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "3d") 
r4D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "4d") 
r5D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "5d") 

MSE0D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "intercept", sklearn = False) 
MSE1D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "1d", sklearn = False) 
MSESimple = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "simple", sklearn = False) 
MSE3D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "3d", sklearn = False) 
MSE4D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "4d", sklearn = False) 
MSE5D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "5d", sklearn = False) 


-round(pricesBMReal - p0D,3)
-round(pricesBMReal - p1D,3)
-round(pricesBMReal - pSimple,3)
-round(pricesBMReal - p3D,3) 
-round(pricesBMReal - p4D,3) 
-round(pricesBMReal - p5D,3)

np.round(r0D,2)
np.round(r1D,2)
np.round(rSimple,2)
np.round(r3D,2) 
np.round(r4D,2) 
np.round(r5D,2)


print(f"{MSE0D:.2e}")
print(f"{MSE1D:.2e}")
print(f"{MSESimple:.2e}")
print(f"{MSE3D:.2e}")
print(f"{MSE4D:.2e}")
print(f"{MSE5D:.2e}")


sr0D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "intercept",sklearn = True) 
sr1D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "1d",sklearn = True) 
srSimple = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,sklearn = True) 
sr3D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "3d",sklearn = True) 
sr4D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "4d",sklearn = True) 
sr5D = LSM.getFirstRegParams(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "5d",sklearn = True) 

sp0D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "intercept", sklearn = True) 
sp1D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "1d", sklearn = True) 
spSimple = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1, sklearn = True) 
sp3D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "3d", sklearn = True) 
sp4D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "4d", sklearn = True) 
sp5D = LSM.priceDefault(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "5d", sklearn = True) 

sMSE0D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "intercept", sklearn = True) 
sMSE1D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "1d", sklearn = True) 
sMSESimple = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "simple", sklearn = True) 
sMSE3D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "3d", sklearn = True) 
sMSE4D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "4d", sklearn = True) 
sMSE5D = LSM.getFirstRegMSE(putOpt, sim, dt= delta, r=r, K=1.1,regMethod = "5d", sklearn = True) 

-round(pricesBMReal - sp0D,4)
-round(pricesBMReal - sp1D,4)
-round(pricesBMReal - spSimple,4)
-round(pricesBMReal - sp3D,4) 
-round(pricesBMReal - sp4D,4) 
-round(pricesBMReal - sp5D,4)

np.round(sr0D,2)
np.round(sr1D,2)
np.round(srSimple,2)
np.round(sr3D,2) 
np.round(sr4D,2) 
np.round(sr5D,2)

print(f"{sMSE0D:.2e}")
print(f"{sMSE1D:.2e}")
print(f"{sMSESimple:.2e}")
print(f"{sMSE3D:.2e}")
print(f"{sMSE4D:.2e}")
print(f"{sMSE5D:.2e}")


#%% 3 - plot

PayOff = putOpt.payoff(S=sim[:,-1],K=1.1)
PayOff2 = putOpt.payoff(S=sim[:,-2],K=1.1)
ITM = PayOff2 > 0

plt.plot(sim[ITM,-2],PayOff[ITM]* np.exp(-r * T),'bo', label = "Disc Payoff", markersize = 15)
plt.plot(sim[ITM,-2],np.dot(LSM.getDesign(sim[ITM,-2],'intercept'),sr0D),'o',color=(0.6, 0.5, 0.4)  , label = "Intercept only")
plt.plot(sim[ITM,-2],np.dot(LSM.getDesign(sim[ITM,-2],'1d'),sr1D),'o',color='pink', label = "1 poly")
plt.plot(sim[ITM,-2],np.dot(LSM.getDesign(sim[ITM,-2],'simple'),srSimple),'o',color='black', label = "2 poly")
plt.plot(sim[ITM,-2],np.dot(LSM.getDesign(sim[ITM,-2],'3d'),sr3D),'go', label = "3 poly")
plt.plot(sim[ITM,-2],np.dot(LSM.getDesign(sim[ITM,-2],'4d'),sr4D),'yo', label = "4 poly")
plt.plot(sim[ITM,-2],np.dot(LSM.getDesign(sim[ITM,-2],'5d'),sr5D),'ro', label = "5 poly")

x_vec = np.linspace(sim[ITM,-2].min(), sim[ITM,-2].max(), 200)
plt.plot(x_vec, putOpt.payoff(S=x_vec,K=1.1), 'k-', linewidth=1, label="Payoff Now")

plt.legend(loc = "upper center",ncol=2)
plt.plot(x_vec, np.dot(LSM.getDesign(x_vec, 'intercept'), sr0D), '-', color = (0.6, 0.5, 0.4)   ,linewidth=1, label="Intercept line")
plt.plot(x_vec, np.dot(LSM.getDesign(x_vec, '1d'), sr1D), '-', color = "pink" ,linewidth=1, label="1 poly line")
plt.plot(x_vec, np.dot(LSM.getDesign(x_vec, 'simple'), srSimple), 'k-', linewidth=1, label="2 poly line")
plt.plot(x_vec, np.dot(LSM.getDesign(x_vec, '3d'), sr3D), 'g-', linewidth=1, label="3 poly line")
plt.plot(x_vec, np.dot(LSM.getDesign(x_vec, '4d'), sr4D), 'y-', linewidth=1, label="4 poly line")
plt.plot(x_vec, np.dot(LSM.getDesign(x_vec, '5d'), sr5D), 'r-', linewidth=1, label="5 poly line")


plt.xlabel("X")
plt.ylabel("$")
plt.title("LSM - In Sample Predictions")
#%% 4
import Helpers.LongstaffSchwartz as LSM
from Helpers.SimulateProcess import Simulate_GBM
from Options.Put import put
from scipy.stats import kurtosis, skew, skewnorm

putOpt = put()
delta = 1
T = 3
r = 0.06
stdReal = 0.15 
m = 8
n = 10000

prices = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=n, sigma = stdReal,density = True,antihetic = False)

fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.hist(prices, bins=100, color='skyblue', edgecolor='black', alpha=0.7,density = True)
axs.axvline(x=np.mean(prices), color='blue', linestyle='--', linewidth=2,label = "Mean")
axs.axvline(x=0.1144343, color='red', linestyle='--', linewidth=2,label = "LS Price")
axs.axvline(x=0.0926, color='green', linestyle='--', linewidth=2,label = "Real Price")
axs.set_title("Distribution of $\sigma_{0.15}$ price estimates")
axs.set_xlabel("$\pi_{BM}$")
axs.set_ylabel("Density")
axs.legend(loc="upper left")
axs.text(0.88, 0.95, f'Mean: {np.mean(prices):.4f}\nStd: {np.std(prices,ddof=1):.6f}\nKurtosis: {kurtosis(prices):.4f}\nSkewness: {skew(prices):.4f}', 
            transform=axs.transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

a, loc, scale = skewnorm.fit(prices)
x_vec = np.linspace(np.min(prices), np.max(prices), 1000)
p = skewnorm(a,loc,scale).pdf(x_vec)
axs.plot(x_vec,p,color = "black" ,label = "Skew Normal Fit")

axs.legend(loc="upper left")

np.mean(np.array(prices) >= 0.1144343)

np.mean(prices)
np.std(prices,ddof = 1)

#%% 5
import Helpers.LongstaffSchwartz as LSM
from Helpers.SimulateProcess import Simulate_GBM
from Options.Put import put
from scipy.stats import kurtosis, skew, skewnorm

putOpt = put()
delta = 1
T = 3
r = 0.06
stdReal = 0.15 
m = 8
n = 10000
B = [[ 2.037512, - 3.335443, 1.356457],
     [ - 1.069988, 2.983411, - 1.813576]]

prices = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=n, sigma = stdReal,B=B,antihetic = False)


fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.hist(prices, bins=100, color='skyblue', edgecolor='black', alpha=0.7,density = True)
axs.axvline(x=np.mean(prices), color='blue', linestyle='--', linewidth=2,label = "Mean")
axs.axvline(x=0.1144343, color='red', linestyle='--', linewidth=2,label = "LS Price")
axs.axvline(x=0.0926, color='green', linestyle='--', linewidth=2,label = "Real Price")
axs.set_title("Distribution of $\sigma_{0.15}$ price estimates with fixed regression parameters")
axs.set_xlabel("$\pi_{BM}$")
axs.set_ylabel("Density")
axs.text(0.88, 0.95, f'Mean: {np.mean(prices):.4f}\nStd: {np.std(prices,ddof=1):.6f}\nKurtosis: {kurtosis(prices):.4f}\nSkewness: {skew(prices):.4f}', 
            transform=axs.transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

a, loc, scale = skewnorm.fit(prices)
x_vec = np.linspace(np.min(prices), np.max(prices), 1000)
p = skewnorm(a,loc,scale).pdf(x_vec)
axs.plot(x_vec,p,color = "black" ,label = "Skew Normal Fit")

axs.legend(loc="upper left")

np.mean(np.array(prices) >= 0.1144343)
np.mean(prices)
np.std(prices,ddof = 1)

#%% 5 - plot 2
import Helpers.LongstaffSchwartz as LSM
from Helpers.SimulateProcess import Simulate_GBM
from Options.Put import put
from scipy.stats import kurtosis, skew, skewnorm

putOpt = put()
delta = 1
T = 3
r = 0.06
stdReal = 0.15 
m = 8
n = 10000

simulation = Simulate_GBM(mu=r,delta=delta,T=3,S0=1,m=int(10000000*delta/3),sigma=stdReal, antihetic = False).T
p, CF, B = LSM.priceDefault(putOpt, simulation, dt = delta, r = r
                                  , K=1.1,sklearn=True, moreInfo = True)

prices = LSM.priceDefaultMultiple(putOpt, Simulate_GBM, dt = delta, r = r
                                  , K=1.1,T=T,S0=1,m=m,n=n, sigma = stdReal,B=B,antihetic = False)


fig, axs = plt.subplots(1, 1, figsize=(15, 5))

axs.hist(prices, bins=100, color='skyblue', edgecolor='black', alpha=0.7,density = True)
axs.axvline(x=np.mean(prices), color='blue', linestyle='--', linewidth=2,label = "Mean")
axs.axvline(x=0.1144343, color='red', linestyle='--', linewidth=2,label = "LS Price")
axs.axvline(x=0.0926, color='green', linestyle='--', linewidth=2,label = "Real Price")
axs.set_title("Distribution of $\sigma_{0.15}$ price estimates with improved regression parameters")
axs.set_xlabel("$\pi_{BM}$")
axs.set_ylabel("Density")
axs.text(0.88, 0.95, f'Mean: {np.mean(prices):.4f}\nStd: {np.std(prices,ddof=1):.6f}\nKurtosis: {kurtosis(prices):.4f}\nSkewness: {skew(prices):.4f}', 
            transform=axs.transAxes, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

a, loc, scale = skewnorm.fit(prices)
x_vec = np.linspace(np.min(prices), np.max(prices), 1000)
p = skewnorm(a,loc,scale).pdf(x_vec)
axs.plot(x_vec,p,color = "black" ,label = "Skew Normal Fit")

axs.legend(loc="upper left")
