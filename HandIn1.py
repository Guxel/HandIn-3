"""
Code script for FinKont2
Hand-In 1
Gustav Dyhr
msh606
2025-02-20
"""
#%% Global
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 800    
#plt.rcParams["figure.constrained_layout.use"] = True
"""
sys.exit()
#%% 2.b, plot 1
from Options.BachelierCall import bachelierCall
Bachlier = bachelierCall(precision = 1000)

df = pd.DataFrame(index = np.arange(1,1501)/10)
df.index.name = 'Strike'
df['opt price'] = Bachlier.price(S=100,K=df.index.to_numpy(),sigma=15,T=0.25)

plt.plot(df['opt price'].astype(np.float64))
plt.title("Bachelier Call Option")
plt.ylabel("$\pi_t$",labelpad=1)
plt.xlabel("K")

#%% plot 2
from Options.Call import call
BS = call(precision = 1000)

df['imp vol'] = BS.invertVol(price = df['opt price'].to_numpy(),S=100,K = df.index.to_numpy(), T=0.25, r = 0, convergence = 1.5e-16)

plt.plot(df['imp vol'].astype(np.float64))
plt.title("Bachelier Implied Volatility $(S_0=100)$")
plt.ylabel("$\sigma^{imp}$",labelpad=1)
plt.xlabel("K")

#%% plot 3
df['BS 0.01'] = BS.price(sigma=0.01,S=100,K = df.index.to_numpy(), T=0.25, r = 0) - df['opt price']
df['BS 0.2'] = BS.price(sigma=0.2,S=100,K = df.index.to_numpy(), T=0.25, r = 0) - df['opt price']
df['BS 0.75'] = BS.price(sigma=0.75,S=100,K = df.index.to_numpy(), T=0.25, r = 0) - df['opt price']

plt.plot(df['BS 0.01'], label = '$\sigma=0.01$')
plt.plot(df['BS 0.2'],color= 'red', label = '$\sigma=0.2$')
plt.plot(df['BS 0.75'],color='green', label = '$\sigma=0.75$')

plt.title("Naive BS price differences")
plt.legend()
plt.ylabel("$\pi^{real}-\pi^{BS}$",labelpad=1)
plt.xlabel("K")

#%% plot 4
df50 = pd.DataFrame(index = df.index.values)
df50.index.name = 'Strike'
df50['opt price'] = Bachlier.price(S=50,K=df.index.to_numpy(),sigma=15,T=0.25)
df50['imp vol'] = BS.invertVol(price = df50['opt price'].to_numpy(),S=50,K = df.index.to_numpy(), T=0.25, r = 0)

plt.plot(df['imp vol'].astype(np.float64),label = "$S_0=100$")
plt.plot(df50['imp vol'].astype(np.float64), label = "$S_0=50$",color = 'red')
plt.title("Bachelier Implied Volatility Comparison")
plt.legend()
plt.ylabel("$\sigma^{imp}$",labelpad=1)
plt.xlabel("K")

#%%3.b
from Options.QuantoPut import quantoPut
from DiscreteHedge import SimulateBadForeignDeltaHedge
from Helpers.SimulateProcess import simulate_brownian
S0 = K = 30000
Y = X0 = 1/100
T = 2
rD = 0.03
rF = 0
sigmaX = np.array((0.1,0.02))
sigmaF = np.array((0,0.25))

opt = quantoPut()

#%% plot 1, naive hedge
n = 1000
m = 100000
brownian2D = simulate_brownian(T/n,n,m,p=2)
hedge = SimulateBadForeignDeltaHedge(brownian2D,opt, X0, S0, T, rD,rF, sigmaF,sigmaX, K=K, Y=Y)


plt.plot(hedge['asset value'],hedge['hedging pf value'], 'o',markersize=0.75,color = 'black',label="Hedge Attempt")
plt.plot(hedge['asset value'],hedge['option payoff'], 'o',markersize=0.75,color = 'blue', label="Option Payoff")
plt.ylabel('USD')
plt.xlabel('$S_T^J$ in Yen')
plt.title(f"Naive Hedge Attempt $(n={n})$")
plt.legend()
plt.tight_layout()
plt.show()

#%% plot 2, errors by n
m = 5000
ns = 2 ** np.arange(0, 16)
errors = pd.DataFrame(columns=['n','std'])
for n in ns:
    print(n)
    brownian2D = simulate_brownian(T/n,n,m,p=2)
    hedge = SimulateBadForeignDeltaHedge(brownian2D,opt, X0, S0, T, rD,rF, sigmaF,sigmaX, K=K, Y=Y)
    errors.loc[n] = [n,np.sqrt(np.var(hedge['discounted hedge error']))]
 
plt.plot(errors['n'],errors['std'])
plt.ylabel('$\sigma_{\epsilon}$')
plt.xlabel('$n$')
plt.yscale('log')
plt.xscale('log')
plt.title("Discounted Hedge error for $n$ approaching $\infty$")
plt.show()
"""
#%%3.c
from Options.QuantoPut import quantoPut
from Helpers.SimulateProcess import simulate_brownian
S0 = K = 30000
Y = X0 = 1/100
T = 2
rD = 0.03
rF = 0
sigmaX = np.array((0.1,0.02))
sigmaF = np.array((0,0.25))

opt = quantoPut()

from DiscreteHedge import SimulateGoodForeignDeltaHedge
#%% plot 1, good hedge
n = 1000
m = 100000
brownian2D = simulate_brownian(T/n,n,m,p=2)
hedge = SimulateGoodForeignDeltaHedge(brownian2D,opt, X0, S0, T, rD,rF, sigmaF,sigmaX, K=K, Y=Y)


plt.plot(hedge['asset value'],hedge['hedging pf value'], 'o',markersize=0.75,color = 'black',label="Hedge Attempt")
plt.plot(hedge['asset value'],hedge['option payoff'], 'o',markersize=0.75,color = 'blue', label="Option Payoff")
plt.ylabel('USD')
plt.xlabel('$S_T^J$ in Yen')
plt.title(f"Good Hedge Attempt $(n={n})$")
plt.legend()
plt.tight_layout()
plt.show()

#%% plot 2, errors by n
from sklearn.linear_model import LinearRegression

m = 5000
ns = 2 ** np.arange(0, 16)
errors = pd.DataFrame(columns=['n','std'])
for n in ns:
    print(n)
    brownian2D = simulate_brownian(T/n,n,m,p=2)
    hedge = SimulateGoodForeignDeltaHedge(brownian2D,opt, X0, S0, T, rD,rF, sigmaF,sigmaX, K=K, Y=Y)
    errors.loc[n] = [n,np.sqrt(np.var(hedge['discounted hedge error']))]
 
    
reg = LinearRegression().fit(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['std']))
pred  = np.exp(reg.predict(np.log(errors['n'].values.reshape(-1, 1))))
Rsq = reg.score(X = np.log(errors['n'].values).reshape(-1, 1),y = np.log(errors['std']))

plt.plot(errors['n'],errors['std'], label = 'Observed $\sigma_{\epsilon}$')
plt.plot(errors['n'],pred,linestyle = '--', color = 'red', label = 'Regression on $\sigma_{\epsilon}$')
plt.ylabel('$\sigma_{\epsilon}$')
plt.xlabel('$n$')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.text(1, 0.5,f"Slope = {reg.coef_} \n$R^2 = $ {Rsq}",fontsize=12, color='black')
plt.title("Good Hedge - Discounted Hedge error for $n$ approaching $\infty$")
plt.show()
