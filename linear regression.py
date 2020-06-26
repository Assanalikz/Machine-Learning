import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import math


x = np.array([5,3,4,10,15])
y = np.array([25,20,21,35,38])

sumx = sum(x)
sumy = sum(y)
sqsumofx = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2
sqsumofy = y[0] ** 2 + y[1] ** 2 + y[2] ** 2 + y[3] ** 2 + y[4] ** 2
xy = x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3] + x[4] * y[4]
correlation = (5 * xy - sumx * sumy) / (math.sqrt(5 * sqsumofx - sumx ** 2) * math.sqrt(5 * sqsumofy - sumy ** 2))
m = (5 * xy - sumx * sumy) / (5 * sqsumofx - sumx ** 2)
b = (sumy / 5) - m * (sumx / 5)
#Graphing Linear Regression 
# y = mx + b 

print('the correlation is', correlation)
print(m)
print(b)
linreg = LinearRegression()
x = x.reshape(-1, 1)
linreg.fit(x, y)
y_pred = linreg.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color = 'black')
plt.show()
print(linreg.coef_)
print(linreg.intercept_)