from svmutil import *
y, x = svm_read_problem('svmguide1'); # 3089 samples

import random as rd
SEED = 42 # change it freely
rd.seed(SEED)
rd.shuffle(x)
rd.seed(SEED)
rd.shuffle(y)

TS = 2500 # train samples
x_train, y_train = x[:TS], y[:TS]
x_test, y_test = x[TS:], y[TS:]

def minmax(X):
	M = [-1e18] * len(X[0])
	m = [1e18] * len(X[0])
	for x in X:
		for i in range(1,len(X[0])+1):
			M[i-1] = max(M[i-1],x[i])
			m[i-1] = min(m[i-1],x[i])
	return m, M

def minmaxScaling(X, m, M):
	for i in range(0, len(X)):
		for j in range(1, len(X[0])+1):
			X[i][j] = (X[i][j]-m[j-1]) / (M[j-1]-m[j-1]) 

m, M = minmax(x_train)
minmaxScaling(x_train, m, M)
minmaxScaling(x_test, m, M)

prob = svm_problem(y_train, x_train)
param = svm_parameter('-s 0 -c 1 -t 2') # RBF kernel
m = svm_train(prob, param) # a ctype pointer

svm_predict(y[TS:], x[TS:], m)
