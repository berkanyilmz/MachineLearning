import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

datas = pd.read_csv('wine.csv') 
X = datas.iloc[:, 0:13].values
y = datas.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components=2)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#before PCA
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#after PCA
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

#without PCA
cm = confusion_matrix(y_test, y_pred)
print(cm)

#with PCA
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)






