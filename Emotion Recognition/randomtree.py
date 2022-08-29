# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 08:28:42 2022

@author: berga
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

train_data = pd.read_csv('training.csv')
'''
0 -> Sadness
1 -> Joy
2 -> Love
3 -> Anger
4 -> Fear
'''
x = train_data.iloc[:,:1].values
y = train_data.iloc[:,1:].values

le = LabelEncoder()
x[:,0] = le.fit_transform(train_data.iloc[:,:1])

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


#y = scaler.fit_transform(y)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500, max_depth=4, bootstrap=True,
                             random_state=18, max_features=1)
rfc.fit(x_train, y_train)
'''
plt.scatter(x_train, y_train, color='red')
plt.scatter(x_test, rf_reg.predict(x_test), color='blue')
plt.show()
'''
'''
string = np.array([['i now feel compromised and skeptical of the value of every unit of work i put in']])
string[:,0] = le.fit_transform(string[:,0])
string = scaler.fit_transform(string)
print('str : ', string)
print('Predict : ', rf_reg.predict([[-0.868716]]))
'''

y_pred = rfc.predict(x_test)
ascr = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(ascr)