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

def EMA(Closes, L:int):
    n0 = np.size(Closes)
    n = n0 - L + 1
    M = np.zeros(n)
    M[0] = np.mean(Closes[:L])
    for i in range(1, n):
        M[i] = M[i-1] + (2/(L + 1))*((Closes[i+L-1]) - M[i-1])
    return M

def MACD(Closes, L1:int, L2:int, Ls:int):
    Slow = EMA(Closes, L1)
    Fast = EMA(Closes, L2)
    maCD = Fast[-np.size(Slow):] - Slow
    S = EMA(maCD, Ls)
    H = maCD[-np.size(S):] - S
    return maCD, S, H

_, _, _, Closes, _ = Fetch('ETH')

maCD, S, H = MACD(Closes, 26, 12, 5)

Z = np.zeros_like(H)

Time1 = np.arange(1, 1001, 1)
Time2 = np.arange(1+np.size(Closes)-np.size(maCD), 1001, 1)
Time3 = np.arange(1+np.size(Closes)-np.size(S), 1001, 1)

plt.subplot(2,1,1)
plt.semilogy(Time1, Closes, label = 'Close', linewidth = 0.9)
plt.xlabel('Time (Day)')
plt.ylabel('Price ($)')
plt.xlim(1,1001)
plt.title('ETH')
plt.legend()

plt.subplot(2,1,2)
plt.plot(Time2, maCD, label = 'MACD(26, 12, 5)-MACD', linewidth = 0.8)
plt.plot(Time3, S, label = 'MACD(26, 12, 5)-Signal', linewidth = 0.8)
plt.plot(Time3, H, label = 'MACD(26, 12, 5)-Histogram', linewidth = 0.8)
plt.fill_between(Time3, H, Z, where = (H > Z), color = 'lime')
plt.fill_between(Time3, H, Z, where = (H < Z), color = 'crimson')
plt.xlabel('Time (Day)')
plt.ylabel('Value')
plt.xlim(1,1001)
plt.legend()

plt.show()