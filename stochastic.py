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

def SMA(Closes, L:int):
    n0 = np.size(Closes)
    n = n0 - L + 1
    M = np.zeros(n)
    for i in range (0, n):
        M[i] = np.mean(Closes[i:i+L])
    return M

def STC(Closes, Highs, Lows, L:int, S:int = 5):
    n0 = np.size(Closes)
    n = n0 - L + 1
    K = np.zeros(n)
    for i in range(0, n):
        HH = np.max(Highs[i:i + L]) # Highest High
        LL = np.min(Lows[i:i + L])  # Lowest low
        K[i] = 100 * (Closes[i + L - 1] - LL) / (HH - LL)
    D = SMA(K, S)
    return K, D
    
_, Highs, Lows, Closes, _ = Fetch('ETH')

K, D = STC(Closes, Highs, Lows, 20, 4)

Time1 = np.arange(1, 1001, 1)
Time2 = np.arange(1+np.size(Closes)-np.size(K), 1001, 1)
Time3 = np.arange(1+np.size(Closes)-np.size(D), 1001, 1)

plt.subplot(2,1,1)
plt.semilogy(Time1, Closes, label = 'Close', linewidth = 0.9)
plt.xlabel('Time (Day)')
plt.ylabel('Price ($)')
plt.xlim(0,1001)
plt.title('ETH')
plt.legend()

plt.subplot(2,1,2)
plt.plot(Time2, K, label = 'STC(20)-K', linewidth = 0.8)
plt.plot(Time3, D, label = 'STC(20)-D', linewidth = 0.8)
plt.plot([1, 1000], [+30, +30], c = 'crimson', linewidth = 0.6)
plt.plot([1, 1000], [+70, +70], c = 'crimson', linewidth = 0.6)
plt.xlabel('Time (Day)')
plt.ylabel('Value')
plt.xlim(0,1001)
plt.legend()

plt.show()