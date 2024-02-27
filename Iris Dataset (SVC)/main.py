import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

#Veri yükleme
dosya = pd.read_csv('iris.csv')

x = dosya.iloc[:,1:5] #bağımsız değişken
y = dosya.iloc[:,5:]

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

ac = accuracy_score(y_test, y_pred)
print(ac) #0.98