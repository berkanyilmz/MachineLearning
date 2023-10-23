import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

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