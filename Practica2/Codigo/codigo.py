#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:59:08 2018

@author: francisco
"""

import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(30)


def simula_unif(N=2, dims=2, size=(0, 1)):
    m = np.random.uniform(low=size[0], 
                          high=size[1], 
                          size=(N, dims))    
    return m

# Función que devuelve un vector de forma <size> con valores aleatorios 
# extraídos de una normal de media <media> y varianza <sigma>.
# Se corresponde con simula_gaus(N, dim, sigma), en caso de llamarla
# con simula_gaus((N, dim0, dim1, ...), sigma).
def simula_gaus(size, sigma, media=None):
    media = 0 if media is None else media
    
    if len(size) >= 2:
        N = size[0]
        size_sub = size[1:]
        
        out = np.zeros(size, np.float64)
        
        for i in range(N):
            out[i] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=size_sub)
    
    else:
        out = np.random.normal(loc=media, scale=sigma, size=size)
    
    return out

# Función que devuelve los parámetros a y b de una recta aleatoria,
# y = a*x + b, tal que dicha recta corta al cuadrado definido por 
# por los puntos (intervalo[0], intervalo[0]) y 
# (intervalo[1], intervalo[1]).
def simula_recta(intervalo=(-1,1), ptos = None):
    if ptos is None: 
        m = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    
    a = (m[0,1]-m[1,1])/(m[0,0]-m[1,0]) # Calculo de la pendiente.
    b = m[0,1] - a*m[0,0]               # Calculo del termino independiente.
    
    return a, b
	
'''
    Transforma los parámetros de una recta 2d a los coeficientes de w.
    a: Pendiente de la recta.
    b: Término independiente de la recta.
'''
def line2coef(a, b):
    w = np.zeros(3, np.float64)
    #w[0] = a/(1-a-b)
    #w[2] = (b-b*w[0])/(b-1)
    #w[1] = 1 - w[0] - w[2]
    
    #Suponemos que w[1] = 1
    w[0] = -a
    w[1] = 1.0
    w[2] = -b
    
    return w	
	
'''
    Pinta los datos con su etiqueta y la recta definida por a y b.
    X: Datos (Intensidad promedio, Simetría).
    y: Etiquetas (-1, 1).
    a: Pendiente de la recta.
    b: Término independiente de la recta.
'''
def plot_datos_recta(X, y, a, b, title='Point clod plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    w = line2coef(a, b)
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = grid.dot(w)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
                      vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$w^tx$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white', label='Datos')
    ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    ax.legend()
    plt.title(title)
    plt.show()

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la complejidad de H y el ruido ------------------#
#-------------------------------------------------------------------------------#

print ('\nEJERCICIO SOBRE SOBRE LA COMPLEJIDAD DE H Y EL RUIDO')

#------------------------------Ejercicio 1 -------------------------------------#

print ('\nEJERCICIO 1\n')

# a) simula_unif N=50, dim=2, rango=(-50, 50)

print ('SIMULA_UNIF')
	
X = simula_unif(N=50, dims=2, size=(-50, 50))
plt.scatter(X[:, 0], X[:, 1])
plt.show()

input("\n--- Pulsar enter para continuar ---\n")

# b) simula_gaus	

print ('\nSIMULA_GAUS')

X = simula_gaus(size=(50,2),sigma=(5,7))
plt.scatter(X[:, 0], X[:, 1])
plt.show()

input("\n--- Pulsar enter para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

print ('\nEJERCICIO 2\n')

def f(X,a,b):
	return X[:, 1]-a*X[:, 0]-b

# simulamos una muestra de puntos 2D 
X = simula_unif(N=100, dims=2, size=(-50, 50))

# simulamos una recta
a,b = simula_recta()

# agregamos las etiquetas usando el signo de la funcion f
y = np.sign(f(X,a,b))

# a) Dibujar gráfica

print ('\na) Puntos etiquetados junto con la recta usada para etiquetar')

plot_datos_recta(X, y, a, b)

input("\n--- Pulsar enter para continuar ---\n")

# b) 

print ('\nb) Introducimos ruido en las etiquetas')

# vector de indices correspondientes a los 'y' con etiqueta 1
ind_pos = []
# vector de indices correspondientes a los 'y' con etiqueta -1
ind_neg = []

# vector y de etiquetas que vamos a modificar
y_mod = np.array(y)

for i in range(0,y.size):
	if y[i] == 1:
		ind_pos.append(i)
	else:
		ind_neg.append(i)

ind_pos = np.array(ind_pos)
ind_neg = np.array(ind_neg)
	
# permutamos los indices
np.random.shuffle(ind_pos)
np.random.shuffle(ind_neg)

# Modificamos 10% de los positivos
for i in range(0,ind_pos.size//10):
	y_mod[ind_pos[i]] = -1

# Modificamos 10% de los negativos
for i in range(0,ind_neg.size//10):
	y_mod[ind_neg[i]] = 1	

plot_datos_recta(X, y_mod, a, b)

input("\n--- Pulsar enter para continuar ---\n")

#------------------------------Ejercicio 3 -------------------------------------#


def plot_datos_funcion(X, y, fz, title='Point clod plot',
					    xaxis='x axis', yaxis='y axis'):
	#Preparar datos
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.002
	
	#Generar grid de predicciones
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+
	border_xy[0]+0.001:border_xy[0], 
	min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = fz(grid)
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
	    
	#Plot
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',
	vmin=-1, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label('$f(x, y)$')
	ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
	cmap="RdYlBu", edgecolor='white', label='Datos')

	ax.contour(xx, yy, pred_y, [0])
	ax.set(
	xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
	ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
	xlabel=xaxis, ylabel=yaxis)
	ax.legend()
	
	plt.title(title)
	plt.show()
	
	
def f1(X):
	return (X[:,0]-10)**2+(X[:,1]-20)**2-400

def f2(X):
	return 0.5*(X[:,0]+10)**2+(X[:,1]-20)**2-400
	
def f3(X):
	return 0.5*(X[:,0]-10)**2-(X[:,1]+20)**2-400

def f4(X):
	return X[:,1]-20*X[:,0]**2-5*X[:,0]+3

print('\nf(x,y) = (x-10)^2+(y-20)^2-400\n')
plot_datos_funcion(X, y_mod, f1)

print('\nf(x,y) = 0.5*(x+10)^2+(y-20)^2-400\n')
plot_datos_funcion(X, y_mod, f2)

print('\nf(x,y) = 0.5*(x-10)^2-(y+20)^2-400\n')
plot_datos_funcion(X, y_mod, f3)

print('\nf(x,y) = y-20x^2-5x+3\n')
plot_datos_funcion(X, y_mod, f4)

input("\n--- Pulsar enter para continuar ---\n")

#-------------------------------------------------------------------------------#
#---------------------------- Modelos Lineales----------------------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#

def ajusta_PLA(datos,label,max_iter,vini):	
	w = vini
	x = np.concatenate((datos, np.ones((datos.shape[0], 1), np.float64)), axis=1)
	iters = 0
	while iters < max_iter:			
		stop = True
		iters+=1
		for j in range(0,label.size):
			# si no tienen el mismo signo actualizamos w
			if np.sign(w.dot(x[j])) != label[j]:
				w = w + label[j]*x[j]
				stop = False
		# si no se ha actualizado w en ninguna de las iteraciones paramos
		if stop:
			break				
	return w, iters	
	
# a)
	
print ('\nPLA datos apartado 2a\n')

vini = np.zeros(X.shape[1]+1, np.float64)
w, iters = ajusta_PLA(X, y, 500, vini)
print (' Iteraciones con vector cero:', iters)

suma = 0

for i in range(0,10):
	vini = simula_unif(N=1, dims=3, size=(0,1))	
	w, iters = ajusta_PLA(X, y, 500, vini)
	suma = suma + iters 
	
print (' Media iteraciones vector num aleatorios: ', suma/10)


# b)
	
print ('\n\nPLA datos apartado 2b\n')

vini = np.zeros(X.shape[1]+1, np.float64)
w, iters = ajusta_PLA(X, y_mod, 500, vini)
print (' Iteraciones con vector cero:', iters)

suma = 0

for i in range(0,10):
	vini = np.random.uniform(0, 1, X.shape[1]+1)	
	w, iters = ajusta_PLA(X, y_mod, 500, vini)
	suma = suma + iters 
	
print (' Media iteraciones vector num aleatorios: ', suma/10)
	
input("\n--- Pulsar enter para continuar ---\n")

#----------------------------- Ejercicio 2 -------------------------#
	
print('\nEJERCICIO 2\n')

def sigmoide(x):
	return 1/(np.exp(-x)+1)

# Regresion logistica Gradiente Descendente Estocastico
def rl_sgd(X, y, max_iters, tam_minibatch, lr = 0.01, epsilon = 0.01):		
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	w = np.zeros(len(x[0]), np.float64)	
	index = np.array(range(0,x.shape[0]))
	tam = tam_minibatch
	
	for k in range(0,max_iters):
		w_old = np.array(w)
		# permutamos los indices para el minibatch
		np.random.shuffle(index)		
		for j in range(0,w.size):			
			
			suma = np.sum(-y[index[0:tam:1]]*x[index[0:tam:1],j]*
			(sigmoide(-y[index[0:tam:1]]*(x[index[0:tam:1]].dot(w_old)))))
			
			w[j] -= lr*suma
			
		if np.linalg.norm(w-w_old) < epsilon:
			break
		
	return w

a, b = simula_recta(intervalo=(0,2))	
X = simula_unif(N=100, dims=2, size=(0,2))
y = np.sign(f(X,a,b))

    
def Err(X,y,w):	
	tam = X.shape[0]	
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	return (1.0/tam)*np.sum(np.log(1+np.exp(-y[0:tam:1]*(x[0:tam:1].dot(w)))))
	
def reetiquetar(y):
	y_ = np.array(y)

	for i in range(0, y_.size):
		if y_[i] == -1:
			y_[i] = 0
			
	return y_
	
def error_acierto(X,y,w):		
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	tam = y.size
	suma = 0
	
	for i in range(0,tam):
		if np.abs(sigmoide(x[i].dot(w))-y[i]) > 0.5:
			suma += 1
			
	return suma/tam	
	
w = rl_sgd(X ,y, 1000, 64)

y_ = reetiquetar(y)

print ('\nEin:',error_acierto(X,y_,w))	
	
X_test = simula_unif(N=2000, dims=2, size=(0,2))
y_test = np.sign(f(X_test,a,b))	

y_test_ = reetiquetar(y_test)

print ('\nEout:',error_acierto(X_test,y_test_,w))


input("\n--- Pulsar enter para continuar ---\n")

#---------------------------- BONUS -------------------------------#

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 4 or datay[i] == 8:
			if datay[i] == 8:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y
	
# Algoritmo pseudoinversa	
def pseudoinversa(X, y):	
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	# Obtenemos la descomposicion en valores singulares
	u,d,vt = np.linalg.svd(x)	
	# calculamos la inversa de d
	d_inv = np.zeros((d.size, d.size), np.float64)	
	for i in range(0,d.size):
		d_inv[i,i] = 1/d[i]
		
	v = vt.transpose()
	# Calculo de la pseudoinversa de x
	x_inv = v.dot(d_inv).dot(d_inv).dot(v.transpose()).dot(x.transpose())
	w = x_inv.dot(y)
	return w

def error_acierto2(X,y,w):		
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	tam = y.size
	suma = 0
	
	for i in range(0,tam):
		if np.sign(x[i].dot(w)) != y[i]:
			suma += 1
			
	return suma/tam	

def PLA_pocket(datos,label,max_iter,vini):	
	w = vini
	x = np.concatenate((datos, np.ones((datos.shape[0], 1), np.float64)), axis=1)
	# vamos a ir almacenando el mejor w encontrado junto al error que 
	# se obtiene con dicho w
	mejor = w
	error_mejor = error_acierto2(datos,y,w)
	
	for i in range(0,max_iter):			
		stop = True
		# Igual que en el PLA
		for j in range(0,label.size):
			if np.sign(w.dot(x[j])) != label[j]:
				w = w + label[j]*x[j]
				stop = False
		if stop:
			break
		else:
			error_actual = error_acierto2(datos,y,w)
			# si el error actual mejora al mejor actualizamos
			if error_actual < error_mejor:
				error_mejor = error_actual
				mejor = np.array(w)			
				
	return mejor
	

print('BONUS\n')

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

vini = pseudoinversa(x,y)
w = PLA_pocket(x, y, 500, vini)

a = -w[0]/w[1]
b = -w[2]/w[1]

# Dibujamos la recta obtenida con PLA_pocket junto con los datos de entrenamiento
plot_datos_recta(x, y, a, b, 'PLA_pocket training', 'Intensidad', 'Simetria')

# Dibujamos la recta obtenida con PLA_pocket junto con los datos para el test
plot_datos_recta(x_test, y_test, a, b, 'PLA_pocket test', 'Intensidad', 'Simetria')

# Calculamos Ein y Etest
ein = error_acierto2(x,y,w)
etest = error_acierto2(x_test,y_test,w)

print ('\nBondad del resultado:\n')
print ('Ein: ',ein)
print ('\nEtest: ', etest)

cotaein = ein + np.sqrt((8/y.size)*np.log(4*((2*y.size)**3 + 1)/0.05))
cotaetest = etest + np.sqrt((8/y_test.size)*np.log(4*((2*y_test.size)**3)/0.05))

print ('\nCotas sobre el valor de Eout\n')
print ('Cota basada en Ein:', cotaein)
print ('Cota basada en Etest:', cotaetest)



