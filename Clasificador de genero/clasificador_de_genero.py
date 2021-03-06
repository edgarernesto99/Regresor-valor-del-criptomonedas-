# -*- coding: utf-8 -*-
"""clasificador de genero.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_csSS1yViHdIwoLLtw0fsAi3yXtufjST
"""

#Primero importo las librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from skimage import io
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#Obtener datos
from google.colab import files
files.upload()

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/drive/MyDrive/age_gender.csv")
data

#Aqui los datos junto con su categoria (image_name, categoria)
images = np.array(([data['pixels'][i].split() 
                    for i in range(data.shape[0])]), dtype=np.float)

n_samples = data.shape[0]

#Obtener datos
x = np.asanyarray(images)
y = np.asanyarray(data['gender'], dtype=np.int)[:n_samples]

x.shape

#Separar datos de prueba y de entrenamiento
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

#Generacion del modelo
model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64)),
    ('mlp', MLPClassifier(alpha=10, max_iter=5000))])
    #('svm', svm.SVC(gamma=0.1))])
    #("KNN", KNeighborsClassifier(1))])
    #('DT', DecisionTreeClassifier(max_depth=50))])
    #("GP", GaussianProcessClassifier(1.0 * RBF(1.0)))])

#Entrenar el modelo con los datos de entrenamiento
model.fit(xtrain, ytrain)

# #*************************************************************
#RESULTADOS
print ("Train: ", model.score(xtrain, ytrain))
print ("Test: ", model.score(xtest, ytest))

# ypred = model.predict(xtest)

# print ('Classification report: \n', metrics.classification_report(ytest, ypred))
# print ('Confusion matrix: \n', metrics.confusion_matrix(ytest, ypred))

# sample = np.random.randint(xtest.shape[0])
# plt.imshow(xtest[sample].reshape((28,28)), cmap=plt.cm.gray)
# plt.title('Prediccion: %i' % ypred[sample])

#Exportar red neuronal
import joblib

filename = 'modelo_clasificador.sav'
joblib.dump(model, filename)
model = joblib.load('modelo_clasificador.sav')

files.download('modelo_clasificador.sav')