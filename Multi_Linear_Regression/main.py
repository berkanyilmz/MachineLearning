# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:43:13 2022

@author: berga
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read data
tennis = pd.read_csv('tennis.csv')

outlook = tennis.iloc[:, 0:1].values
windy = tennis.iloc[:, 3:4].values
play = tennis.iloc[:, -1].values
temp = tennis.iloc[:, 1:2].values
humidity = tennis.iloc[:, 2:3].values

# Label Encoding
labelEncoder = preprocessing.LabelEncoder()

outlook[:,0] = labelEncoder.fit_transform(outlook)
windy = labelEncoder.fit_transform(windy)
play = labelEncoder.fit_transform(play)


# One Hot Encoding
ohe = preprocessing.OneHotEncoder()

outlook = ohe.fit_transform(outlook).toarray()

# Making dataframe
outlook = pd.DataFrame(data=outlook, index=range(14), columns=['overcast',
                                                               'rain',
                                                               'sunny'])
windy = pd.DataFrame(data=windy, index=range(14), columns=['windy'])
temp = pd.DataFrame(data=temp, index=range(14), columns=['temp'])
humidity = pd.DataFrame(data=humidity, index=range(14), columns=['humidity'])
# inputs

x = pd.concat([outlook, windy, temp, humidity], axis=1)

# train and test split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    play,
                                                    test_size=0.33,
                                                    random_state=0)

#fit
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(y_pred)




















