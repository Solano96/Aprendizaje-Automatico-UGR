#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc


def pairs(data, y, names):
	d = data.shape[1]
	fig, axes = plt.subplots(nrows=d, ncols=d, sharex='col', sharey='row')
	fig.set_size_inches(18.5, 10.5)
	for i in range(d):
		for j in range(d):
			ax = axes[i,j]
			if i == j:
				ax.text(0.5, 0.5, names[i], transform=ax.transAxes,
				horizontalalignment='center', verticalalignment='center', fontsize=9)
			else:
				ax.scatter(data[:,j], data[:,i], c=y)
	plt.show()
	input("Pulsa enter para continuar.")

y = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=1, dtype=str)
X = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=range(2, 32))

le = preprocessing.LabelEncoder()
y = le.fit(y).transform(y)

pairs(X[:,0:9], y, ['Radio', 'Textura', 'Perímetro', 'Superficie', 'Suavidad', 'Compacidad', 'Concavidad', 'Ptos conc.', 'Simetría', 'Dim. fractal'])

# separar en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state = 0)

# preprocesado
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# probando hiperparametros
c = np.float_power(10, range(-7,7))
param_dist = {'C': c}
lr = GridSearchCV(LogisticRegression(), cv = 10, param_grid=param_dist)

# Ajustamos el modelo a partir de los datos
lr.fit(X_train, y_train)	

scores = lr.cv_results_['mean_test_score']

plt.plot(c, scores)
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('score')
plt.show()

print ("\nMejor valor de C: ", lr.best_params_['C'])
# Calculamos el score con dicho ajuste para test
predictions_train = lr.predict(X_train)	
score_train = lr.score(X_train, y_train)
	
# Calculamos el score con dicho ajuste para test
predictions_test = lr.predict(X_test)	
score_test = lr.score(X_test, y_test)

print('Valor de acierto con el mejor c sobre el conjunto train: ', score_train)
print('Valor de acierto con el mejor c sobre el conjunto test: ', score_test)
input("Pulsa enter para continuar.")

#Matriz de confusión
print ("\nMatriz de confusión:")
cm = metrics.confusion_matrix(y_test, predictions_test)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score_test)
plt.title(all_sample_title, size = 10);
plt.show()
input("Pulsa enter para continuar.")


# Curva roc
print ("\nCurva ROC:")
y_pred_rf = lr.predict_proba(X_test)[:, 1]
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

