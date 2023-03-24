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

def ATR(Closes, Highs, Lows, L:int):
    n0 = np.size(Closes)
    n = n0 - 1
    TR = np.zeros(n)
    for i in range(0, n):
        TR[i] = max(Highs[i+1],Closes[i]) - min(Lows[i+1],Closes[i])
    aTR = SMMA(TR, L)
    return aTR

def ATR2(Closes, Highs, Lows, L:int):
    n0 = np.size(Closes)
    n = n0 - 1
    TR = np.zeros(n)
    for i in range(0, n):
        TR[i] = (max(Highs[i+1],Closes[i]) - min(Lows[i+1],Closes[i]))/Closes[i+1]
    aTR = SMMA(TR, L)
    return aTR

_, Highs, Lows, Closes, _ = Fetch('ETH')

aTR = ATR(Closes, Highs, Lows, 10)

Time1 = np.arange(1, 1001, 1)
Time2 = np.arange(1+np.size(Closes)-np.size(aTR), 1001, 1)

plt.subplot(2,1,1)
plt.semilogy(Time1, Closes, label = 'Close', linewidth = 0.9)
plt.xlabel('Time (Day)')
plt.ylabel('Price ($)')
plt.xlim(0,1001)
plt.title('ETH')
plt.legend()

plt.subplot(2,1,2)
plt.plot(Time2, aTR, label = 'ATR(10)', linewidth = 0.8)
plt.plot([1, 1000], [0, 0], c = 'gray', linewidth = 0.6, linestyle = '--')
plt.xlabel('Time (Day)')
plt.ylabel('Value')
plt.xlim(0,1001)
plt.legend()

plt.show()