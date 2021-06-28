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

#Aqui los datos junto con su categoria (image_name, categoria)
data = pd.read_csv('oxford-102-flowers/test.txt', delimiter=' ',
                   header=None).values

n_samples = data.shape[0]
images = []
#sizes = np.zeros((n_samples,2))

for i in range(n_samples):
    img = io.imread('oxford-102-flowers/'+data[i,0])/255
    img = resize(img, (64,64))
    #grey = np.sum(img, axis=2)/3
    #sizes[i,0], sizes[i,1]= grey.shape[0], grey.shape[1]
    images.append(img.ravel())
    if i%500 == 0:
        print(i)

images = np.array(images)

#Obtener datos
x = np.asanyarray(images)
y = np.asanyarray(data[:,1], dtype=np.int)[:n_samples]

# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1)

model = Pipeline([
    #('scaler', StandardScaler()),
    ('pca', PCA(n_components=128)),
    #('mlp', MLPClassifier(alpha=10, max_iter=50000))])
    #('svm', svm.SVC(gamma=0.01))])
    #("KNN", KNeighborsClassifier(1))])
    #('DT', DecisionTreeClassifier(max_depth=50))])
    ("GP", GaussianProcessClassifier(1.0 * RBF(1.0)))])

model.fit(x, y)

print ("Train: ", model.score(x, y))

#*************************************************************

#Aqui los datos junto con su categoria (image_name, categoria)
data2 = pd.read_csv('oxford-102-flowers/train.txt', delimiter=' ',
                   header=None).values

xtest = []
n_samples2 = 200
for i in range(n_samples2):
    img = io.imread('oxford-102-flowers/'+data2[i,0])/255
    img = resize(img, (64,64))
    #grey = np.sum(img, axis=2)/3
    #sizes[i,0], sizes[i,1]= grey.shape[0], grey.shape[1]
    xtest.append(img.ravel())
    
xtest = np.array(xtest)
ytest = np.asanyarray(data2[:,1], dtype=np.int)[:n_samples2]
print ("Test: ", model.score(xtest, ytest))

# ypred = model.predict(xtest)

# print ('Classification report: \n', metrics.classification_report(ytest, ypred))
# print ('Confusion matrix: \n', metrics.confusion_matrix(ytest, ypred))

# sample = np.random.randint(xtest.shape[0])
# plt.imshow(xtest[sample].reshape((28,28)), cmap=plt.cm.gray)
# plt.title('Prediccion: %i' % ypred[sample])




