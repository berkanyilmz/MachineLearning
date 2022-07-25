#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#read data
data = pd.read_csv('Position_Salaries.csv')

level = data.iloc [:,1:2]
salary = data.iloc[:,2:]

#dataframe to numpy array
levelX = level.values
salaryY = salary.values


#Linear Regression
linReg = LinearRegression()
linReg.fit(levelX, salaryY)

#Polynomial Regression
polyReg = PolynomialFeatures(degree=4)
xPoly = polyReg.fit_transform(levelX)
linReg2 = LinearRegression()
linReg2.fit(xPoly, salary)

#Data visulation
plt.scatter(levelX, salaryY, color='red')
plt.plot(levelX, linReg2.predict(polyReg.fit_transform(levelX)), color='blue')
plt.show()

print(linReg2.predict(polyReg.fit_transform(levelX)))