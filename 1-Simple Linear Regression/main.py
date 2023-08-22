import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Veri yükleme
dosya = pd.read_csv('satislar.csv')

ay = dosya[['Aylar']]
satis = dosya[['Satislar']]

x_train, x_test, y_train, y_test = train_test_split(ay, satis, test_size=0.33, random_state=0)
'''
x_train -> Eğitim için kullanılacak kısım
x_test -> Eğitilen kısmı test etmek için kullanılacak kısım
y_train -> Eğitilecek kısım
y_test -> Eğitilen kısmı test etmek için kullanılacak kısım
'''
print(x_train)
print(x_test)
print(y_train)
print(y_test)

'''
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
#Model inşası
lr = LinearRegression()
lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_test.values, lr.predict(x_test), color='red')
plt.show()