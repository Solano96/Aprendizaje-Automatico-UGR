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

# Leemos los datos
y = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=1, dtype=str)
X = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=range(2, 32))

le = preprocessing.LabelEncoder()
y = le.fit(y).transform(y)

# Separamos los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state = 0)

clf = RandomForestClassifier(random_state=10)

# Entrenamos con los datos del train
clf.fit(X_train, y_train)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

features_to_use = indices[0:14]

# No quedamos con las caracteristicas mas importantes
X_train = X_train[:,features_to_use]
X_test = X_test[:,features_to_use]

# Probamos hiperparametros
fit_rf = RandomForestClassifier(max_features = 'sqrt', random_state=10)
estimators = range(10,200,10)
param_dist = {'n_estimators': estimators}
clf= GridSearchCV(fit_rf, cv = 10, param_grid=param_dist, n_jobs = 3)

# Entrenamos el clasificador
clf.fit(X_train, y_train)

# Grafica ntree - test error
scores = clf.cv_results_['mean_test_score']
plt.plot(estimators, 1-scores)
plt.xlabel('num tree')
plt.ylabel('test error')
plt.show()

# Mejor parametro
best_param = clf.best_params_['n_estimators']
print ("Mejor valor para n_estimators: ", best_param)


predictions_train = clf.predict(X_train)
predictions = clf.predict(X_test)

print ("Train Accuracy :: ", accuracy_score(y_train, predictions_train))
print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
print (" Confusion matrix \n", confusion_matrix(y_test, predictions))

# Curva roc
y_pred_rf = clf.predict_proba(X_test)[:, 1]
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
