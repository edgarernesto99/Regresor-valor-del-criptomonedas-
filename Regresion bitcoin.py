# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-_zZJ3W2g-g9fWV9Xk8dibj4BUXfJehR
"""

#Primero importo las librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

#Obtener archivo
from google.colab import files
files.upload()

data = pd.read_csv('Bitcoin Historical Data.csv')
data

#Datos
x = np.asanyarray(data[['Price']])
x

#Generar dataset con retardos en la señal
#retardo
p=10
data2 = pd.DataFrame(data.Price)
for i in range(1, p+1):
    data2 = pd.concat([data2, data.Price.shift(-i)], axis=1)
data2 = data2[:-p] #Quita los ultimos p dias 

x = np.asanyarray(data2.iloc[:,1:])
y = np.asanyarray(data2.iloc[:,0])

x.shape

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

x

model = svm.SVR(gamma='scale', C=100, epsilon=0.001, kernel='rbf')
model.fit(xtrain, ytrain)

print("Train: ", model.score(xtrain, ytrain))
print("Test: ", model.score(xtest, ytest))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain, ytrain)

print("Train: ", model.score(xtrain, ytrain))
print("Test: ", model.score(xtest, ytest))

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100,20), solver="adam", 
                     activation='relu', batch_size=10) 
model.fit(xtrain, ytrain)

print("Train: ", model.score(xtrain, ytrain))
print("Test: ", model.score(xtest, ytest))

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=10, weights='uniform') 
model.fit(xtrain, ytrain)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=1) 
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)
print(y_pred)

#Exportar red neuronal
import joblib

filename = 'regersor_bitcoin.sav'
joblib.dump(model, filename)

from datetime import datetime
from datetime import timedelta

date = input(str('Fecha a estimar: '))
date = datetime.strptime(date, '%d/%m/%Y')
p=10

df = data[data['Date'] == date.strftime('%d/%m/%Y')]
if df.shape[0] == 0:
  last = 0
  it = int((date - datetime.strptime(data['Date'][0], '%d/%m/%Y')) / timedelta(days=1))
else:
  it = int(1)
  last = data.index[data['Date'] == date.strftime('%d/%m/%Y')][0]

x_pred = np.asanyarray(data['Price'][last:last+p]).reshape((1,-1))

for i in range(it):
  x_new = model.predict(x_pred)
  x_pred = np.insert(x_pred, 0, x_new, axis=1)[:,:p]
x_new

y_pred = np.array(([0,1,2,3,4,5,6,7,8,9]))
plt.plot(y_pred, x_pred.ravel(), "b-")
plt.plot(y_pred, x_pred.ravel(), "ro")
plt.xlabel("$día$", fontsize=18)
plt.ylabel("$Precio$", fontsize=18)
plt.show()