import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

#this one is a small dataset but if its bigger pls split it
#linear regression model
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X, y)

#polynomial
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree = 4) #try different degrees
X_poly = polynomial.fit_transform(X)
linear2 = LinearRegression()
linear2.fit(X_poly, y)

#visualising linear regression results
plt.scatter(X, y, color = 'blue')
plt.plot(X, linear.predict(X), color = 'green')
plt.title('Position v Salary - Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() #results show its not very well adapted

#polynomial visualisation
plt.scatter(X, y, color = 'blue')
plt.plot(X, linear2.predict(polynomial.fit_transform(X)), color = 'green')
plt.title('Position v Salary - Polynomial')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting with linear vs polynomial
linear_predict = linear.predict([[6.5]])
print(linear_predict)

polynomial_predict = linear2.predict(polynomial.fit_transform([[6.5]]))
print(polynomial_predict)
