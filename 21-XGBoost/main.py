import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

datas = pd.read_csv('Churn_Modelling.txt')
X = datas.iloc[:, 3:13].values
y = datas.iloc[:, 13].values

labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])

oneHotEncoder = OneHotEncoder()
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))