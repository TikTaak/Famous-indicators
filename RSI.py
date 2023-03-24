import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Fetch(Symbol:str):
    Columns = ['open (USD)', 'high (USD)', 'low (USD)', 'close (USD)', 'volume']
    DF = pd.read_csv('{}.csv'.format(Symbol), sep = ',', usecols = Columns, header=0)
    O = (DF['open (USD)'].to_numpy())[::-1].copy()
    H = (DF['high (USD)'].to_numpy())[::-1].copy()
    L = (DF['low (USD)'].to_numpy())[::-1].copy()
    C = (DF['close (USD)'].to_numpy())[::-1].copy()
    V = (DF['volume'].to_numpy())[::-1].copy()
    return O, H, L, C, V

def SMMA(Closes, L:int):
    n0 = np.size(Closes)
    n = n0 - L + 1
    M = np.zeros(n)
    M[0] = np.mean(Closes[:L])
    for i in range(1, n):
        M[i] = M[i-1] + ((Closes[i+L-1]) - M[i-1])/L
    return M

def RSI(Closes, L:int):
    n0 = np.size(Closes)
    U = np.zeros(n0 - 1)
    D = np.zeros(n0 - 1)
    for i in range(0, n0 - 1):
        Change = Closes[i+1] - Closes[i]
        U[i] = max(Change, 0)
        D[i] = max(-Change, 0)
    smmaU = SMMA(U, L)
    smmaD = SMMA(D, L)
    RS = smmaU / smmaD
    R = 100 - 100/(1 + RS)
    return R

_, _, _, Closes, _ = Fetch('ETH')

R = RSI(Closes, 20)

Time1 = np.arange(1, 1001, 1)
Time2 = np.arange(1+np.size(Closes)-np.size(R), 1001, 1)

plt.subplot(2,1,1)
plt.semilogy(Time1, Closes, label = 'Close', linewidth = 0.9)
plt.xlabel('Time (Day)')
plt.ylabel('Price ($)')
plt.xlim(0,1001)
plt.title('ETH')
plt.legend()

plt.subplot(2,1,2)
plt.plot(Time2, R, label = 'RSI(20)', linewidth = 0.8)
plt.plot([1, 1000], [30, 30], c = 'crimson', linewidth = 0.6)
plt.plot([1, 1000], [70, 70], c = 'crimson', linewidth = 0.6)
plt.xlabel('Time (Day)')
plt.ylabel('Value')
plt.xlim(0,1001)
plt.ylim(0,100)
plt.legend()

plt.show()