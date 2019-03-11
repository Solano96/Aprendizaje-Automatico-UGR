#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#===================================================================#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold 
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

#===================================================================#

# Funcion para leer los datos
def readData(file_x, file_y):
	x = np.load(file_x)
	y = np.load(file_y)	
	x_ = np.empty(x.shape, np.float64)
	
	for i in range(0,x.shape[0]):
		for j in range(0,x.shape[1]):
			x_[i][j] = np.float64(1.0*x[i][j]/16.0)
	
	return x_, y

# Leemos el conjunto de entrenamiento
X_train, y_train = readData('optdigits_tra_npy/optdigits_tra_X.npy', 
							'optdigits_tra_npy/optdigits_tra_y.npy')

# Visualizamos algunos datos
print('\nVisualización de algunos datos:')

plt.figure(figsize=(10, 5))

for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(np.split(X_train[i], 8), cmap='gray')
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
	
plt.show()

sel = VarianceThreshold(threshold=(0.01))
X_train = sel.fit_transform(X_train)

print('\nLas partes negras corresponten a caracteristicas eliminadas')
plt.imshow(np.split(sel.get_support(), 8), cmap='gray')
plt.show()

poly = PolynomialFeatures(2)
X_train= poly.fit_transform(X_train) 

# KFold
k = 2
kf = StratifiedKFold(n_splits=k)
kf.get_n_splits(X_train,y_train)

mejor_C = 0
mejor_media = 0

x = []
y = []

for i in range(-5,5):			
	suma = 0	
	c = 10**i
	
	for train_index, test_index in kf.split(X_train,y_train):
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		
		lr = LogisticRegression(C=c)
		lr.fit(X_train_, y_train_)	
		suma += lr.score(X_test_, y_test_)
		
	media = 1.0*suma/k
	
	x.append(c)
	y.append(media)
	
	if media > mejor_media:
		mejor_media = media
		mejor_C = c
		
		
print ("\nVariación del score según el valor de C")
plt.plot(x,y)
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('score')
plt.show()

input("\n--- Pulsar enter para continuar ---\n")

X_test, y_test = readData('datos/optdigits_tes_X.npy', 'datos/optdigits_tes_y.npy')	

X_test = sel.transform(X_test)
X_test = poly.fit_transform(X_test) 

lr = LogisticRegression(C=mejor_C)

lr.fit(X_train, y_train)	
predictions = lr.predict(X_test)	
score = lr.score(X_test, y_test)
Etest = 1-mejor_media
print('Etest: 1 - score_validacion = ', 1-mejor_media)
print('Cota de Eout:', Etest+np.sqrt(1/(2*X_train.shape[1]) * np.log(2/0.05) ) )

print('\nScore obtenido en el test:', score)
print('Eout: 1 - score_test = ', 1-score)
print('\nMejor Score obtenido en la validación:', mejor_media)

input("\n--- Pulsar enter para continuar ---\n")

print ("\nMatriz de confusión:")
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
	

