import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('seattle-weather.csv')
#print(dataset)

weather = dataset.loc[:, 'weather'].values

label_encoder = preprocessing.LabelEncoder()
weather = label_encoder.fit_transform(dataset.loc[:, 'weather'])

x = dataset.iloc[:, 1:5]
y = dataset.iloc[:, 5:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
pred = svc.predict([[68, 4, -3, 0.32]])
print('pred : ' + str(pred))