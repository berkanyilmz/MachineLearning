import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_excel('iris.xls')
x = dataset.iloc[:, :4]
y = dataset.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

knn_cm = confusion_matrix(y_test, y_pred)
knn_acc = accuracy_score(y_test, y_pred)
print('KNN Acc : ', knn_acc)


from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

svc_cm = confusion_matrix(y_test, y_pred)
svc_acc = accuracy_score(y_test, y_pred)
print('SVC Acc : ', svca_cc)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

gnb_cm = confusion_matrix(y_test, y_pred)
gnb_acc = accuracy_score(y_test, y_pred)
print('GaussianNB Acc : ', gnb_acc)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)

dtc_cm = confusion_matrix(y_test, y_pred)
dtc_acc = accuracy_score(y_test, y_pred)
print('Decision Tree Classifier Acc : ', dtc_acc)



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='entropy')
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

rfc_cm = confusion_matrix(y_test, y_pred)
rfc_acc = accuracy_score(y_test, y_pred)
print('Random Forest Classifier Acc : ', rfc_acc)