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


_, _, _, Closes, _ = Fetch('ETH')

M = SMA(Closes, 20)

Time1 = np.arange(1, 1001, 1)
Time2 = np.arange(1+np.size(Closes)-np.size(M), 1001, 1)

plt.semilogy(Time1, Closes, label = 'Close', c = 'k', linewidth = 1)
plt.semilogy(Time2, M, label = 'SMA(20)', c = 'r', linewidth = 1.2)
plt.xlabel('Time (Day)')
plt.ylabel('Price ($)')
plt.title('ETH')
plt.legend()
plt.show()
