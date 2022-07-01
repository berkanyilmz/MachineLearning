# (x % 2 == 0) => f(x) = x*3
# (x % 2 != 0) => f(x) = x*2


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data3.csv')
x = data.iloc[:,:1]
y = data.iloc[:,1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.scatter(x_train, y_train, color='red')
plt.plot(x_test.values, lr.predict(x_test), color='blue')
plt.show()
print(lr.predict(x_test))
print('Tahmin : ', lr.predict([[10]]))
