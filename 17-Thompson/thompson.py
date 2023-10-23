import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

#Thompson
N = 10000
d = 10
rewards = [0] * d
clicks = [0] * d
total = 0
choosens = []
ones = [0] * d
zeros = [0] * d
for n in range(1, N):
    ad = 0 #choosen ad
    max_th = 0
    for i in range(0, d):
        randbeta = random.betavariate(ones[i] + 1, zeros[i] + 1)
        if randbeta > max_th:
            max_th = randbeta
            ad = i
            
    choosens.append(ad)
    reward = datas.values[n, ad]
    if reward == 1:
        ones[ad] = ones[ad] + 1
    else:
        zeros[ad] = zeros[ad] + 1

    total = total + reward
    
    
    
print("Total rewards : ", total)
plt.hist(choosens)
    
    
    
    
    
    