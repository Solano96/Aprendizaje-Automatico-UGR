#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:51:59 2018

@author: francisco
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


#------------------------------Ejercicio 1 -------------------------------------#

# Fijamos la semilla
np.random.seed(1)

def E(w): 
	u = w[0]
	v = w[1]
	return (u**3*np.exp(v-2)-4*v**3*np.exp(-u))**2
			 
# Derivada parcial de E respecto de u
def Eu(w):
	u = w[0]
	v = w[1]
	return (2*(u**3*np.exp(v-2)-4*v**3*np.exp(-u))
			 *(3*u**2*np.exp(v-2)+4*v**3*np.exp(-u)))

# Derivada parcial de E respecto de v
def Ev(w):
	u = w[0]
	v = w[1]
	return (2*(u**3*np.exp(v-2)-4*v**3*np.exp(-u))
			*(u**3*np.exp(v-2)-12*v**2*np.exp(-u)))
	
# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

def gd(w, lr, grad_fun, fun, epsilon, max_iters = 100000000):
	it = 0
	# Criterio de parada: numero de iteraciones y 
	# valor de f inferior a epsilon
	while it <= max_iters and fun(w) >= epsilon:
		w = w-lr*grad_fun(w)
		it += 1
		
	return w, it
	
# Ejecutamos el algoritmo con un learning rate de 0.05 y 
# valor de E inferior a 10^-14
w, k = gd(np.array([1.0,1.0]) , 0.05, gradE, E, 10**(-14))

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
print ('Numero de iteraciones: ', k)
input("\n--- Pulsar tecla para continuar ---\n")
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")


#------------------------------Ejercicio 2 -------------------------------------#

pi = math.pi

def f(w):
	x = w[0]
	y = w[1]
	return (x-2)**2+2*(y+2)**2+2*math.sin(2*pi*x)*math.sin(2*pi*y)
	
# Derivada parcial de f respecto de x
def fx(w):
	x = w[0]
	y = w[1]
	return 2*(x-2)+2*math.sin(2*pi*y)*math.cos(2*pi*x)*2*pi

# Derivada parcial de f respecto de y
def fy(w):
	x = w[0]
	y = w[1]
	return 4*(y+2)+2*math.sin(2*pi*x)*math.cos(2*pi*y)*2*pi
	
# Gradiente de f
def gradf(w):
	return np.array([fx(w), fy(w)])
	
# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
def gd_grafica(w, lr, grad_fun, fun, max_iters = 50):
	graf = [fun(w)]
	for k in range(1,max_iters):
		w = w-lr*grad_fun(w)
		graf.insert(len(graf),fun(w))
				
	plt.plot(range(0,max_iters), graf, 'bo')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.show()	

print ('Resultados ejercicio 2\n')
print ('\nGrafica con learning rate igual a 0.01')
gd_grafica(np.array([1.0,1.0]) , 0.01, gradf, f)
print ('\nGrafica con learning rate igual a 0.1')
gd_grafica(np.array([1.0,1.0]) , 0.1, gradf, f)

input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

def gd(w, lr, grad_fun, fun, max_iters = 50):
	for i in range(0,50):
		w = w-lr*grad_fun(w)
		
	return w

print ('Punto de inicio: (2.1, -2.1)\n')
w = gd(np.array([2.1, -2.1]) , 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
w = gd(np.array([3.0, -3.0]) , 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
w = gd(np.array([1.5, 1.5]) , 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
w = gd(np.array([1.0, -1.0]) , 0.01, gradf, f)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y
	
# Funcion para calcular el error
def Err(x,y,w):
	return (1/y.size)*np.linalg.norm(x.dot(w)-y)**2
	
# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):	
	w = np.zeros(len(x[0]), np.float64)
	index = np.array(range(0,x.shape[0]))
	
	for k in range(0,max_iters):
		w_old = w
		for j in range(0,w.size):
			suma = 0			
			# Permutamos los indices para el minibatch
			np.random.shuffle(index)
			tam = tam_minibatch
			# Sumatoria en n de x_index[n]j*(w^Tx_index[n] - y_index[n])
			suma = (np.sum(x[index[0:tam:1],j]*(x[index[0:tam:1]].dot(w_old) 
					- y[index[0:tam:1]])))
			
			w[j] -= lr*(2.0/tam)*suma
		
	return w
	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
	# Obtenemos la descomposicion en valores singulares
	u,d,vt = np.linalg.svd(x)
	d_inv = np.linalg.inv(np.diag(d))
	v = vt.transpose()
	# Calculo de la pseudoinversa de x
	x_inv = v.dot(d_inv).dot(d_inv).dot(v.transpose()).dot(x.transpose())
	w = x_inv.dot(y)
	return w
	
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd(x, y, 0.01, 500, 64)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

# Para el grafico tomamos las columnas 1 y 2 de x que corresponden
# la la intensidad y simetria, y distiguimos las coordenadas por 
# su valor en y, es decir por la clase a la que pertenezcan ({-1,1})
plt.scatter(x[:,1],x[:,2], c=y)
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('SGD')
plt.show()

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

plt.scatter(x[:,1],x[:,2], c=y)
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.title('Pseudoinversa')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))
	
# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

x = simula_unif(1000, 2, 1)
plt.scatter(x[:,0],x[:,1])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# b) Usamos la función f2 para asignar etiquetas a la muestra x e introducimos 
# ruido al 10% de las mismas

# Funcion signo
def sign(x):
	if x >= 0:
		return 1
	return -1

def f2(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

# Array con 10% de indices aleatorios para introducir ruido
p = np.random.permutation(range(0,x.shape[0]))[0:x.shape[0]//10]
# Ordenamos el array
p.sort()
j = 0
y = []

for i in range(0,x.shape[0]):
	# Si i está en p cambiamos el signo
	if i == p[j]:
		j = (j+1)%(x.shape[0]//10)
		y.append(-f2(x[i][0], x[i][1]))
	# En otro caso mantenemos el signo
	else:
		y.append(f2(x[i][0], x[i][1]))

x = np.array(x, np.float64)
y = np.array(y, np.float64)

print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')
print ('con etiquetas y ruido en el 10% de las etiquetas')

plt.scatter(x[:,0],x[:,1], c=y)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# c) Modelo de regresión lineal, estimar Ein con SGD

# Vector de unos para la regresion lineal (1, x0, x1)
a = np.array([np.ones(x.shape[0], np.float64)])
x = np.concatenate((a.T, x), axis = 1)

w = sgd(x, y, 0.01, 1000, 64)

print ('Estimacion del error de ajuste Ein usando SGD')
print ('con 1000 iteraciones y un de minibatch de 64')
print ("\n\nEin: ", Err(x,y,w))
plt.scatter(x[:,1],x[:,2], c=y)
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.axis([-1,1,-1,1])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

print ("Espere 5 segundos...\n")

Ein_media = 0
Eout_media = 0

for k in range(0,1000):
	x = simula_unif(1000, 2, 1)
	# vector de unos para la regresion lineal (1, x0, x1)
	a = np.array([np.ones(x.shape[0], np.float64)])
	x = np.concatenate((a.T, x), axis = 1)
	y = []
	
	# Array con 10% de indices aleatorios para introducir ruido
	p = np.random.permutation(range(0,x.shape[0]))[0:x.shape[0]//10]
	p.sort()
	j = 0
	
	for i in range(0,x.shape[0]):
		if i == p[j]:
			j = (j+1)%(x.shape[0]//10)
			y.append(-f2(x[i][1], x[i][2]))
		else:
			y.append(f2(x[i][1], x[i][2]))
	
	y = np.array(y, np.float64)	
	
	# Solo 10 iteraciones y minibatch de 32 para que la ejecucion
	# no tarde demasiado tiempo
	w = sgd(x, y, 0.01, 10, 32)
	
	# Simulamos los datos del test para despues calcular el Eout
	x_test = simula_unif(1000, 2, 1)	
	x_test = np.concatenate((a.T, x_test), axis = 1)
	
	Ein_media += Err(x,y,w)
	Eout_media += Err(x_test,y,w)

Ein_media /= 1000
Eout_media /= 1000

print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")

#-------------------------------------------------------------------------------#
#---------------------------------- BONUS --------------------------------------#
#-------------------------------------------------------------------------------#

# Segundas derivadas de f para el calculo de la hessiana	

def fxx(w):
	x = w[0]
	y = w[1]
	return 2-2*math.sin(2*pi*y)*2*pi*2*pi*math.sin(2*pi*x)
	
def fxy(w):
	x = w[0]
	y = w[1]
	return 2*math.cos(2*pi*y)*math.cos(2*pi*x)*2*pi*2*pi
	
def fyy(w):
	x = w[0]
	y = w[1]
	return 4-2*math.sin(2*pi*y)*2*pi*2*pi*math.sin(2*pi*x)

# Hessiana de f
def Hf(w):
	return np.array([np.array([fxx(w), fxy(w)]), np.array([fxy(w), fyy(w)])])


# Metodo de Newton
def newtonMethods(w, lr, grad_fun, fun, H, max_iters = 50):
	graf = [fun(w)]	
	for k in range(1,max_iters):		
		# w_{k+1} = w_k - lr*H^-1*gradiente
		w = w - lr*np.linalg.inv(H(w)).dot(grad_fun(w))			
		graf.insert(len(graf),fun(w))		
				
	return w, graf
		
# Ejecutamos el metodo de Newton con punto de inicio (1,1) y lr 0.01
print ('BONUS: metodo de Newton\n')
print ('Punto de inicio: (1.0, 1.0), Learning rate: 0.01\n')		
w, g = newtonMethods(np.array([1.0,1.0]), 0.01, gradf, f, Hf, 50)

# Grafica iteraciones con valores de f obtenidos en el met, Newton
plt.plot(range(0,len(g)), g, 'bo')
plt.xlabel('Iteraciones')
plt.ylabel('f(x,y)')
plt.show()	

input("\n--- Pulsar tecla para continuar ---\n")


# Ejecutamos el metodo de Newton con punto de inicio (1,1)
# y learning rate 0.1
print ('Punto de inicio: (1.0, 1.0), Learning rate: 0.1\n')		
w, g = newtonMethods(np.array([1.0,1.0]), 0.1, gradf, f, Hf)

plt.plot(range(0,len(g)), g, 'bo')
plt.xlabel('Iteraciones')
plt.ylabel('f(x,y)')
plt.show()	

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (2.1, -2.1)\n')
w, g = newtonMethods(np.array([2.1, -2.1]) , 0.01, gradf, f, Hf)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
w, g = newtonMethods(np.array([3.0, -3.0]) , 0.01, gradf, f, Hf)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
w, g = newtonMethods(np.array([1.5, 1.5]) , 0.01, gradf, f, Hf)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
w, g = newtonMethods(np.array([1.0, -1.0]) , 0.01, gradf, f, Hf)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

# Ejecutamos el metodo de Newton con punto de inicio (1,1)
# y learning rate 0.01 y con 2000 iteraciones
print ('Punto de inicio: (1.0, 1.0), Learning rate: 0.01')
print ('2000 iteraciones')		
w, g = newtonMethods(np.array([1.0,1.0]), 0.01, gradf, f, Hf, 2000)

# Grafica iteraciones con valores de f obtenidos en el met, Newton
plt.plot(range(0,len(g)), g, 'bo')
plt.xlabel('Iteraciones')
plt.ylabel('f(x,y)')
plt.show()	


