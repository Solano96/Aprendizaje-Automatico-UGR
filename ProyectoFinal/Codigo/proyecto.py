import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import StratifiedKFold 

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


####################################### MODELO LINEAL ###############################################################
	
y = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=1, dtype=str)
X = np.genfromtxt('datos/wdbc.data', delimiter=',', usecols=range(2, 32))

le = preprocessing.LabelEncoder()

print(y)
y = le.fit(y).transform(y)

print(y)
pairs(X[:,0:9], y, ['Radio', 'Textura', 'Perímetro', 'Superficie', 'Suavidad', 'Compacidad', 'Concavidad', 'Ptos conc.', 'Simetría', 'Dim. fractal'])

print("\nMODELO LINEAL - REGRESION LOGISTICA")
print("---------------------------------------------------\n")

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

input("Pulsa enter para continuar.")

####################################### RANDOM FOREST ###############################################################

print("\nRANDOM FOREST")
print("---------------------------------------------------\n")

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

input("Pulsa enter para continuar.")

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


input("Pulsa enter para continuar.")

# Mejor parametro
best_param = clf.best_params_['n_estimators']
print ("Mejor valor para n_estimators: ", best_param)


predictions_train = clf.predict(X_train)
predictions = clf.predict(X_test)

print ("Train Accuracy :: ", accuracy_score(y_train, predictions_train))
print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
print (" Confusion matrix \n", confusion_matrix(y_test, predictions))

input("Pulsa enter para continuar.")

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

input("Pulsa enter para continuar.")


####################################### SVM ###############################################################

print("\nSVM")
print("---------------------------------------------------\n")

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

input("Pulsa enter para continuar.")

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

input("Pulsa enter para continuar.")

####################################### RED NEURONAL ###############################################################

print("\nRED NEURONAL")
print("---------------------------------------------------\n")

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

input("Pulsa enter para continuar.")

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

input("Pulsa enter para continuar.")

model = MLPClassifier(max_iter=500, hidden_layer_sizes = hls[mejor], random_state = 10)

param = {'alpha': 10.0 ** -np.arange(1, 7)}
mlp = GridSearchCV(model, cv = 10, param_grid=param, n_jobs = 3)
mlp.fit (X_train, y_train)

print ("\nMejor valor de alpha: ", mlp.best_params_['alpha'])

input("Pulsa enter para continuar.")

predictions_train = mlp.predict(X_train)
predictions = mlp.predict(X_test)

print ("Train Accuracy :: ", accuracy_score(y_train, predictions_train))
print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
print (" Confusion matrix \n", confusion_matrix(y_test, predictions))

input("Pulsa enter para continuar.")

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

input("Pulsa enter para continuar.")










