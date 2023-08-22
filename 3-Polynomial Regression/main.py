import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri yükleme
veriler = pd.read_csv("maaslar.csv")

#dataframe dilimleme (slice)
x = veriler.iloc[:, 1:2] #Eğitim seviyesi
y = veriler.iloc[:, 2:] #Maaş bilgisi

#Numpy array dönüşümü
X = x.values
Y = y.values


#lineer regression
#Doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#polynomial regression
#Doğrusal olmayan model oluşturma
#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


#4. dereceden model
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y)


#Görselleştirme
plt.scatter(X,Y,color="red")
plt.plot(x, lin_reg.predict(X),color="blue")
plt.show()

plt.scatter(X, Y, color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.show()

plt.scatter(X, Y, color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color='blue')
plt.show()

#tahminler
print(lin_reg.predict([[11]]))
