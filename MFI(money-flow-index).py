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

def tp(Closes, Highs, Lows):
    TP = (Closes + Highs + Lows)/3
    return TP

def SMA(Closes, L:int):
    n0 = np.size(Closes)
    n = n0 - L + 1
    M = np.zeros(n)
    for i in range (0, n):
        M[i] = np.mean(Closes[i:i+L])
    return M

def MFI(Closes, Highs, Lows, Volumes, L:int):
    TP = tp(Closes, Highs, Lows)
    n0 = np.size(TP)
    PMF = np.zeros(n0 - 1)
    NMF = np.zeros(n0 - 1)
    for i in range(0, n0 - 1):
        Change = TP[i+1] - TP[i]
        if Change >= 0:
            PMF[i] = TP[i+1] * Volumes[i+1]
        else:
            NMF[i] = TP[i+1] * Volumes[i+1]
    mPMF = SMA(PMF, L)
    mNMF = SMA(NMF, L)
    MR = mPMF / mNMF
    I = 100 - 100/(1 + MR)
    return I

_, Highs, Lows, Closes, Volumes = Fetch('ETH')

I = MFI(Closes, Highs, Lows, Volumes, 20)

Time1 = np.arange(1, 1001, 1)
Time2 = np.arange(1+np.size(Closes)-np.size(I), 1001, 1)

plt.subplot(2,1,1)
plt.semilogy(Time1, Closes, label = 'Close', linewidth = 0.9)
plt.xlabel('Time (Day)')
plt.ylabel('Price ($)')
plt.xlim(0,1001)
plt.title('ETH')
plt.legend()

plt.subplot(2,1,2)
plt.plot(Time2, I, label = 'MFI(20)', linewidth = 0.8)
plt.plot([1, 1000], [+30, +30], c = 'crimson', linewidth = 0.6)
plt.plot([1, 1000], [+70, +70], c = 'crimson', linewidth = 0.6)
plt.xlabel('Time (Day)')
plt.ylabel('Value')
plt.xlim(0,1001)
plt.ylim(0,100)
plt.legend()

plt.show()