import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

#Random Selection
"""
N = 10000
d = 10
total = 0
choosens = []
for n in range(0,N):
    ad = random.randrange(d)
    choosens.append(ad)
    reward = datas.values[n, ad]
    total = total + reward

plt.hist(choosens)
plt.show()
"""

#UCB
N = 10000
d = 10
rewards = [0] * d
clicks = [0] * d
total = 0
choosens = []
for n in range(1, N):
    ad = 0 #choosen ad
    max_ucb = 0
    for i in range(0, d):
        if (clicks[i] > 0):
            avarage = rewards[i] / clicks[i]
            delta = math.sqrt(3/2 * math.log(n)/clicks[i]) 
            ucb = avarage + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
            
    choosens.append(ad)
    clicks[ad] = clicks[ad] + 1
    reward = datas.values[n, ad]
    rewards[ad] = rewards[ad] + reward
    total = total + reward
    
    
    
print("Total rewards : ", total)
    
    
plt.hist(choosens)
    
    
    
    
    
    