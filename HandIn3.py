"""
Code script for FinKont2
Hand-In 3
Gustav Dyhr
msh606
2025-03-28
"""
#%% Global
import pandas as pd
import numpy as np
import sys

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 800    

from Options import Insurance
from ContFin import DiscreteHedge
from Helpers import SimulateProcess
from Helpers import Misc

insuranceOpt = Insurance.insurance()
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

sys.exit()

#%% 2
p = insuranceOpt.price(r=r,sigma=std,A0=A0,S0=S0,S=S0,T=T,t=0,K=K,a=a)
print(p)
delta = 1/252

stockSim = SimulateProcess.Simulate_GBM(T, delta, S0, mu=r, sigma = std,m = 5000, antihetic = True)
hedgeBad = DiscreteHedge.SimulateSimpleDeltaHedge(Misc.select_rows(stockSim, 30), insuranceOpt, T, r, sigma = std, K=K, A0=A0, S0=S0, a=a)
hedgeGood = DiscreteHedge.SimulateSimpleDeltaHedge(stockSim, insuranceOpt, T, r, sigma = std, K=K, A0=A0, S0=S0, a=a)


sns.set_theme()
plt.plot(stockSim[-1],hedgeGood['option payoff'], 'o',markersize=1.5,color = 'blue',label = "Option Payoff")
plt.plot(stockSim[-1],hedgeBad['hedging pf value'], 'o',markersize=0.75,color = 'black', label = "Yearly Hedge Payoff")
plt.plot(stockSim[-1],hedgeGood['hedging pf value'], 'o',markersize=0.75,color = 'red', label = "Daily Hedge Payoff")
plt.ylabel('pfV')
plt.xlabel('S')
plt.title("Discrete Hedge Experiment")
plt.legend()
plt.xlim((-0.1,7.1))
plt.show()

from sklearn.linear_model import LinearRegression
ns = 2 ** np.arange(1, 15)
ns[-1] = 30000
errors = pd.DataFrame(columns=['n','error std'])
for n in ns:
    print(n)
    hedge = DiscreteHedge.SimulateSimpleDeltaHedge(Misc.select_rows(stockSim, n), insuranceOpt, T, r, sigma = std, K=K, A0=A0, S0=S0, a=a)
    errors.loc[n] = [n,np.sqrt(np.var(hedge['discounted hedge error'],ddof=1))]
 
    
reg = LinearRegression().fit(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error std']))
pred  = np.exp(reg.predict(np.log(errors['n'].values.reshape(-1, 1))))
Rsq = reg.score(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error std']))

plt.plot(errors['n'],errors['error std'], label = 'Observed $\sigma_{\epsilon}$')
plt.plot(errors['n'],pred,linestyle = '--', color = 'red', label = 'Regression on $\sigma_{\epsilon}$')
plt.ylabel('$\sigma_{\epsilon}$')
plt.xlabel('$n$')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.text(2, 0.01,f"Slope = {reg.coef_} \n$R^2 = $ {Rsq}",fontsize=12, color='black')
plt.title("Discounted Hedge error for $n$ approaching $\infty$")
plt.show()

#%% 3
from scipy import stats as ss
from Options import Put
putOpt = Put.put()

mu = 0.07

o0 = insuranceOpt.price(r=r,sigma=std,A0=A0,S0=S0,S=S0,T=T,t=0,K=K,a=a)

d1 = (np.log(A0 / K) + (r + 0.5 * (std * a) ** 2) * T) / (a*std * np.sqrt(T))
d2 = d1 - a * std * np.sqrt(T)
o1 = np.exp(-r*T)*K*ss.norm.cdf(-d2+(r-mu)/std*np.sqrt(T)) - np.exp(a*(mu-r)*T)*A0*ss.norm.cdf(-d1+(r-mu)/std*np.sqrt(T))

gT =  A0 / np.power(S0,a) * np.exp((r+a*(std**2)/2)*(1-a)*T)

d1Star =(np.log(A0 / (K*gT)) + (r + 0.5 * (std * a) ** 2) * T) / (a*std * np.sqrt(T))
d2Star = d1Star - a * std * np.sqrt(T)
o2 = np.exp(-r*T)*K*ss.norm.cdf(-d2Star) - A0/gT *ss.norm.cdf(-d1Star)

o3 = a * putOpt.price(S=S0,T=T,r=r,sigma=std,K=K)

print(o0,o1,o2,o3)

np.exp(-r*T)*ss.norm.cdf(-d2) 
(gT**-1) * insuranceOpt.price(r=r,sigma=std,A0=A0,S0=S0,S=S0,T=T,t=0,K=(K*gT),a=a)

np.exp(-r*T)*ss.norm.cdf(-d2) *K



-1 **-2 * insuranceOpt.price(r=r,sigma=std,A0=A0,S0=S0,S=S0,T=T,t=0,K=(K*1),a=a) + (1 **-1) * np.exp(-r*T)*K*ss.norm.cdf(-d2) 
-gT **-2 * insuranceOpt.price(r=r,sigma=std,A0=A0,S0=S0,S=S0,T=T,t=0,K=(K*gT),a=a) + (gT **-1) * np.exp(-r*T)*K*ss.norm.cdf(-d2Star) 
gT **-2 * A0 *ss.norm.cdf(-d1Star) 

#%% 5
from Options import Put
from Options import Call
putOpt = Put.put()
callOpt = Call.call()

p = insuranceOpt.price(r=r,sigma=std,A0=A0,S0=S0,S=S0,T=T,t=0,K=K,a=a)
gT =  A0 / np.power(S0,a) * np.exp((r+a*(std**2)/2)*(1-a)*T)
Satm = np.power(K/gT,1/a)
Satm
K-gT*(S0 ** a)


dOpt = gT *(a-a**2) #(K**(a-2)) #not part of the option

coeft0 = {'K':K,'S0':S0,'T':T,'r':r,'sigma':std,'A0':A0,'a':a,'S':S0,'t':0}

coefS0 = {'K':K,'S0':S0,'T':0,'r':r,'sigma':std,'A0':A0,'a':a,'S':S0,'t':T}
coefSatm = {'K':K,'S0':S0,'T':0,'r':r,'sigma':std,'A0':A0,'a':a,'S':Satm,'t':T}

dBonds = insuranceOpt.payoff(**coefS0) - insuranceOpt.payoffD(**coefS0) * S0
dStock = insuranceOpt.payoffD(**coefS0) 
dATM = insuranceOpt.payoffDD(**coefSatm)

n = 5
Ks = np.linspace(0, Satm, n+2)[1:-1]
pos = (gT * np.power(Ks,a-2)*(a-a**2)*Satm/(n))
price = np.where(Ks<S0,putOpt.price(K=Ks,S=S0,T=T,r=r,sigma=std),callOpt.price(K=Ks,S=S0,T=T,r=r,sigma=std))
payOffs = np.where(Ks<S0,putOpt.payoff(K=Ks,S=S0,T=T,r=r,sigma=std),callOpt.payoff(K=Ks,S=S0,T=T,r=r,sigma=std))

np.exp(-r*T)*dBonds+ dStock + np.sum(pos*price) + dATM*callOpt.price(K=Satm,S=S0,T=T,r=r,sigma=std)

K-(1-a)*gT* S0 ** a

K*(1-a*gT)-(1-a)*gT


Ss = np.linspace(0, 3, 10000)[1:-1]
a=0.5
Payoffs = insuranceOpt.payoff(K=K,t=T,T=0,S0=S0,A0=A0,S=Ss,a=a,r=r,sigma=std)

#change spanPutOnly to span for calls, stock and bonds

span1 = DiscreteHedge.spanPutOnly(insuranceOpt,area=Ss, n=0, S0=S0, K=K,T=T,r=r,sigma=std,A0=A0,a=a,noBS = True)
span2 = DiscreteHedge.spanPutOnly(insuranceOpt,area=Ss, n=5, S0=S0, K=K,T=T,r=r,sigma=std,A0=A0,a=a,noBS = True)
span3 = DiscreteHedge.spanPutOnly(insuranceOpt,area=Ss, n=250, S0=S0, K=K,T=T,r=r,sigma=std,A0=A0,a=a,noBS = True)

span1['Positions'][0:2]
span2['Positions'][0:2]
span3['Positions'][0:100]
span3['Price']
span3['Fit']

plt.plot(Ss,Payoffs,'--',color = 'red',label = "Insurance Payoff")
plt.plot(Ss,span3['Fit'],label = "Span n=100", color ='yellow')
plt.plot(Ss,span2['Fit'],label = "Span n=5")
plt.plot(Ss,span1['Fit'],label = "Span n=0")
plt.xlim((-0.1,3.1))
plt.ylim((-0.25,1.9))

plt.ylabel('f(S)')
plt.xlabel('S')
plt.title("Spanning The Insurance Contract")
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
ns = 2 ** np.arange(0, 17)
errors = pd.DataFrame(columns=['n','error Square','Price Error'])
for n in ns:
    print(n)
    span = DiscreteHedge.spanPutOnly(insuranceOpt,area=Ss, n=n, S0=S0, K=K,T=T,r=r,sigma=std,A0=A0,a=a,noBS = True)
    errors.loc[n] = [n,np.sqrt(np.sum(np.power(Payoffs-span['Fit'],2))),np.abs(p-span['Price'])]


reg = LinearRegression().fit(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error Square']))
pred  = np.exp(reg.predict(np.log(errors['n'].values.reshape(-1, 1))))
Rsq = reg.score(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error Square']))

reg2 = LinearRegression().fit(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['Price Error']))
pred2  = np.exp(reg2.predict(np.log(errors['n'].values.reshape(-1, 1))))
Rsq2 = reg2.score(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['Price Error']))


plt.plot(errors['n'],errors['error Square'], label = 'Replication Errors')
plt.plot(errors['n'],pred,linestyle = '--', color = 'red')
plt.plot(errors['n'],errors['Price Error'], label = 'Price Errors')
plt.plot(errors['n'],pred2,linestyle = '--')
plt.ylabel('$\sqrt{\sum\epsilon^2}$')
plt.xlabel('$n$')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.text(20, 0.9,f"Slope = {reg.coef_} \n$R^2 = $ {Rsq}",fontsize=12, color='black')
plt.text(1, 0.00001,f"Slope = {reg2.coef_} \n$R^2 = $ {Rsq2}",fontsize=12, color='black')
plt.title("Slope error for $n$ approaching $\infty$")
plt.show()

#%% 7
from Options.NaiveHestonPut import naiveHestonPut
putOpt = naiveHestonPut(fixD=True) 

phiLim = 15
n = 1000000
phis = np.tile(np.linspace(0, phiLim,n+2)[1:-1].reshape(-1,1),2)
    

Kc= K
Kc2 = 1e-5

char = putOpt.characteristic(phis, S=A0,K=Kc,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
inner = np.real((np.exp(-1j*phis*np.log(Kc))*char)/(1j*phis))

char2 = putOpt.characteristic(phis, S=A0,K=Kc2,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
inner2 = np.real((np.exp(-1j*phis*np.log(Kc2))*char)/(1j*phis))
    
import matplotlib.pyplot as plt
    
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6),sharex=True)

axes[0,0].plot(phis[:, 0], inner[:, 0])
axes[1,0].plot(phis[:, 0], inner[:, 1], color='red')
axes[0,1].plot(phis[:, 0], inner2[:, 0])
axes[1,1].plot(phis[:, 0], inner2[:, 1], color='red')
    
axes[1,0].set_xlabel("$\phi_K$")
axes[1,1].set_xlabel("$\phi_K$")
    
axes[0,0].set_ylabel("$j=1$")
axes[1,0].set_ylabel("$j=2$")

axes[0,0].set_title("$K=e^{rT}$")
axes[0,1].set_title("$K=1e-05$")
    
fig.suptitle("Real Values of Naive Heston Implementation with flipped $d$")
plt.show()

from Options.NaiveHestonPut import naiveHestonPut
phiLim = 15
n = 50000
putOpt = naiveHestonPut(fixD=True) 
r=0.02
putOpt.price(phiLim=phiLim,n=n,  S=A0,K=K,T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)

from Options.LakeHestonPut import lakeHestonPut
putOptLake = lakeHestonPut()
price = putOptLake.price(N=500,batchTol=1e-6,S=A0,K=np.array(((1,K,3),(0.1,0.2,0.3))),T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
print(price)


#%% 8
from Options.LakeHestonPut import lakeHestonPut
N = 1000
Ss = np.linspace(0, 3, 10000)[1:-1]
a=0.5
Payoffs = insuranceOpt.payoff(K=K,t=T,T=0,S0=S0,A0=A0,S=Ss,a=a,r=r,sigma=std)

putOptLake = lakeHestonPut() 

spanHest = DiscreteHedge.spanPutOnly(insuranceOpt,area=Ss, n=10000,putOpt=putOptLake, S0=S0, K=K,T=T,r=r,sigma=std,A0=A0,a=a,V=(V0),kappa=kappa,epsilon=epsilon,theta=theta,lambd=lambd,rho=rho,noBS = True)
spanHest['Price']



from sklearn.linear_model import LinearRegression
ns = 2 ** np.arange(0, 17)
errors = pd.DataFrame(columns=['n','error Square','Price'])
for n in ns:
    print(n)
    span = DiscreteHedge.spanPutOnly(insuranceOpt,putOpt = putOptLake,area=Ss, n=n, S0=S0, K=K,T=T,r=r,sigma=std,A0=A0,a=a,noBS = True,V=(V0),kappa=kappa,epsilon=epsilon,theta=theta,lambd=lambd,rho=rho)
    errors.loc[n] = [n,np.sqrt(np.sum(np.power(Payoffs-span['Fit'],2))),span['Price']]

reg = LinearRegression().fit(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error Square']))
pred  = np.exp(reg.predict(np.log(errors['n'].values.reshape(-1, 1))))
Rsq = reg.score(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error Square']))


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6),sharex=True)

axes[0].plot(errors['n'],errors['error Square'])
axes[0].plot(errors['n'],pred,linestyle = '--', color = 'red')

axes[1].plot(errors['n'],errors['Price Error'])

axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_ylabel('$\sqrt{\sum\epsilon^2}$')

axes[1].set_xscale('log')
axes[1].set_xlabel('$n$')
axes[1].set_ylabel('$\pi^{SPAN}$')

axes[0].text(1, 0.001,f"Slope = {reg.coef_} \n$R^2 = $ {Rsq}",fontsize=12, color='black')
axes[1].text(10000, 0.209,f"Price = {round(errors.iloc[-1,-1],4)}",fontsize=12, color='black')
axes[0].set_title("Spanning The Heston Model")
plt.show()


a=1
price = putOptLake.price(N=50,S=A0,K=np.array((1,2,K)),T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
print(price)

spanHest = DiscreteHedge.spanPutOnly(insuranceOpt,area=Ss, n=10000,putOpt=putOptLake, S0=S0, K=K,T=T,r=r,sigma=std,A0=A0,a=a,V=(V0),kappa=kappa,epsilon=epsilon,theta=theta,lambd=lambd,rho=rho,noBS = True)
spanHest['Price']
spanHest['Positions']

#%% 9
from Options.LakeHestonPut import lakeHestonPut
import pandas as pd
putOptLake = lakeHestonPut() 

delt = putOptLake.delta(lazyAlpha=True,N=500,batchTol=1e-6,S=A0,K=np.array(((1,K,3),(0.1,0.2,0.3))),T=T,V=(a*a*V0),kappa=kappa,epsilon=epsilon*a,theta=a*a*theta,lambd=lambd,rho=rho,r=r)
print(delt)

dt = 1/(252*24)
n = 5000
batchSize=10000

hestonProcess = SimulateProcess.simulateHestonDynamicsBatched(T, dt, S0, r, V0, theta, kappa, epsilon, rho,n=n,batchSize=batchSize)


hedgeBad = DiscreteHedge.SimulateSimpleDeltaHedge(Misc.select_rows(hestonProcess[:,:,0], 30), putOptLake, T, r, sigma = Misc.select_rows(hestonProcess[:,:,1], 30), K=K, S0=S0, theta=theta,epsilon=epsilon,kappa=kappa,lambd=lambd,rho=rho,N=100,batchTol=1e-6,lazyAlpha=True)
hedgeGood = DiscreteHedge.SimulateSimpleDeltaHedge(Misc.select_rows(hestonProcess[:,:,0], 30*252), putOptLake, T, r, sigma = Misc.select_rows(hestonProcess[:,:,1], 30*252), K=K, S0=S0, theta=theta,epsilon=epsilon,kappa=kappa,lambd=lambd,rho=rho,N=100,batchTol=1e-6,lazyAlpha=True)


sns.set_theme()
plt.plot(hestonProcess[-1,:,0],hedgeBad['option payoff'], 'o',markersize=1.5,color = 'blue',label = "Put Payoff",zorder=3)
plt.plot(hestonProcess[-1,:,0],hedgeBad['hedging pf value'], 'o',markersize=0.75,color = 'black', label = "Yearly Hedge Payoff",zorder=2)
plt.plot(hestonProcess[-1,:,0],hedgeGood['hedging pf value'], 'o',markersize=0.75,color = 'red', label = "Daily Hedge Payoff",zorder=1)
plt.plot(pd.Series(hedgeGood['hedging pf value'],hestonProcess[-1,:,0])
         .sort_index().rolling(window=50, min_periods=1).mean(),color = 'yellow',label = "Daily Hedge Payoff MA"
         )
plt.ylabel('pfV')
plt.xlabel('S')
plt.title("Discrete Hedge Experiment - Heston")
plt.legend()
plt.xlim((-0.3,6))
plt.show()



from sklearn.linear_model import LinearRegression
ns = 2*(1.6 ** np.arange(1, 16)).astype(int)
ns = np.hstack((ns[0:5], ns[6:]))
errors = pd.DataFrame(columns=['n','error std'])
for n in ns:
    print(n)
    hedge = DiscreteHedge.SimulateSimpleDeltaHedge(Misc.select_rows(hestonProcess[:,:,0], n), putOptLake, T, r, sigma = Misc.select_rows(hestonProcess[:,:,1], n), K=K, S0=S0, theta=theta,epsilon=epsilon,kappa=kappa,lambd=lambd,rho=rho,N=100,batchTol=1e-6,lazyAlpha=True)
    errors.loc[n] =[n,np.sqrt(np.sum(np.power(hedge['discounted hedge error'],2)))]

errors.loc[30] = [30,np.sqrt(np.sum(np.power(hedgeBad['discounted hedge error'],2)))]
errors.loc[30*252] = [30*252,np.sqrt(np.sum(np.power(hedgeGood['discounted hedge error'],2)))]

errors = errors.sort_index()

reg = LinearRegression().fit(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error std']))
pred  = np.exp(reg.predict(np.log(errors['n'].values.reshape(-1, 1))))
Rsq = reg.score(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['error std']))

plt.plot(errors['n'],errors['error std'])
plt.plot(errors['n'],pred,linestyle = '--', color = 'red')
plt.ylabel('$\sqrt{\sum\epsilon^2}$')
plt.xlabel('$n$')
plt.yscale('log')
plt.xscale('log')
plt.text(2, 6,f"Slope = {reg.coef_} \n$R^2 = $ {Rsq}",fontsize=12, color='black')
plt.title("Discounted Hedge error for $n$ approaching $\infty$ - Heston Delta Hedge")
plt.show()



