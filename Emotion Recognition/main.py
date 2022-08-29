# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

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


knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
true = 0
total = 0
for i in range(0,6):
    for j in range(0,6):
        if i is j:
            true = true + cm[i,j]
        total = total + cm[i,j]
print('total : ', total)
print('true : ', true)
accuracyRate = true / total
print ('Accuracy Rate : {}'.format(accuracyRate))            






