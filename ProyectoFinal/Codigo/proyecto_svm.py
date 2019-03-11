#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn import svm

y = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=1, dtype=str)
X = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=range(2, 32))

le = preprocessing.LabelEncoder()
y = le.fit(y).transform(y)

# separar en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state = 0)

# preprocesado
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# Validacion y regularizacion
c_range = np.float_power(10, range(-7,7))
degree_range = list(range(1,5))
param = dict(degree=degree_range, C=c_range)
svmachine=svm.SVC(kernel='poly', probability=True)
clf = GridSearchCV(svmachine, cv = 10, param_grid=param)

# Ajustamos el modelo a partir de los datos
clf.fit(X_train, y_train)

# Dibujamos las gráficas en función de C y degree
params = clf.cv_results_['params']
scores = clf.cv_results_['mean_test_score']

plt.plot(c_range, scores[0::4], 'r-', label='grado 1')
plt.plot(c_range, scores[1::4], 'b-', label='grado 2')
plt.plot(c_range, scores[2::4], 'g-', label='grado 3')
plt.plot(c_range, scores[3::4], 'y-', label='grado 4')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('score')
plt.legend()
plt.show()

# Calculamos el score con dicho ajuste para test
predictions_train = clf.predict(X_train)	
score_train = clf.score(X_train, y_train)
	
# Calculamos el score con dicho ajuste para test
predictions_test = clf.predict(X_test)
score_test = clf.score(X_test, y_test)

print('\nMejor valor de C y mejor grado: ', clf.best_params_)
print('Número de vectores de soporte para cada clase: ', clf.best_estimator_.n_support_)
print('Valor de acierto con el mejor c sobre el conjunto train: ', score_train)
print('Valor de acierto con el mejor c sobre el conjunto test: ', score_test)
input('Pulsa enter para continuar.')


#Matriz de confusión
print ('\nMatriz de confusión:')
cm = metrics.confusion_matrix(y_test, predictions_test)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score_test)
plt.title(all_sample_title, size = 10);
plt.show()
input('Pulsa enter para continuar.')

# Curva roc
print ('\nCurva ROC:')
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