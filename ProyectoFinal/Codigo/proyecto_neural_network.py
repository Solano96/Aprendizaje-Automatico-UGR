#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold 

# Leemos los datos
y = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=1, dtype=str)
X = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=range(2, 32))

le = preprocessing.LabelEncoder()
y = le.fit(y).transform(y)

# Separamos los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state = 0)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test) 

hls = []
for j in range(0,3):
	for i in range(1,50,5):
		v = []
		for k in range(0,j+1):
			v.append(i)
		hls.append(v)

print("hidden_layer_sizes:\n", hls)

# KFold
k = 5
kf = StratifiedKFold(n_splits=k)
kf.get_n_splits(X_train,y_train)

mejor = 0
mejor_media = 0

for i in range(0,len(hls)):			
	suma = 0		
	for train_index, test_index in kf.split(X_train,y_train):
		X_train_, X_test_ = X_train[train_index], X_train[test_index]
		y_train_, y_test_ = y_train[train_index], y_train[test_index]
		
		mlp = MLPClassifier(max_iter=500, hidden_layer_sizes = hls[i], random_state = 10)
		mlp.fit(X_train_, y_train_)	
		suma += mlp.score(X_test_, y_test_)		
		
	media = 1.0*suma/k	
	
	if media > mejor_media:
		mejor_media = media
		mejor = i

print("Mejor hidden_layer_sizes: ", hls[mejor])

model = MLPClassifier(max_iter=500, hidden_layer_sizes = hls[mejor], random_state = 10)

param = {'alpha': 10.0 ** -np.arange(1, 7)}
mlp = GridSearchCV(model, cv = 10, param_grid=param, n_jobs = 3)
mlp.fit (X_train, y_train)

print ("\nMejor valor de alpha: ", mlp.best_params_['alpha'])

predictions_train = mlp.predict(X_train)
predictions = mlp.predict(X_test)

print ("Train Accuracy :: ", accuracy_score(y_train, predictions_train))
print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
print (" Confusion matrix \n", confusion_matrix(y_test, predictions))

# Curva roc
print ("\nCurva ROC:")
y_pred_rf = mlp.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)  
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

