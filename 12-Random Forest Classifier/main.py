import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


dataframe = pd.read_csv('veriler.txt')
x = dataframe.iloc[:, 1:4]
y = dataframe.iloc[:, 4:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

rfc = RandomForestClassifier(criterion='entropy')
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)