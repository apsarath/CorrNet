__author__ = 'Sarath'

import numpy
import theano
import theano.tensor as T
from numpy import *


'''
Optimization methods : Gradient Descent, Classical Momentum, Nesterov Accelerated Gradient
                       RMSProp, Adagrad, Adadelta

Gradients can be clipped between -1 to 1.
'''

def get_optimizer(optimization, l_rate, eps = 1e-8, decay = 0.9, momentum = 0.9, clipping = False ):
	if(optimization == "sgd"):
		return Optimizer(l_rate, clipping)
	elif(optimization == "cm"):
		return CM(l_rate, momentum, clipping)
	elif(optimization == "nag"):
		return NAG(l_rate, momentum, clipping)
	elif(optimization == "rmsprop"):
		return RMSProp(l_rate, eps, decay, clipping)
	elif(optimization == "adagrad"):
		return Adagrad(l_rate, eps, clipping)
	elif(optimization == "adadelta"):
		return Adadelta(l_rate, eps, decay, clipping)


class Optimizer(object):
	'''   Simple Gradient Descent          '''

	def __init__(self,l_rate,clipping=False):
		self.l_rate = theano.shared(numpy.asarray(l_rate, dtype = float32), borrow = True)
		self.wlist = {}
		self.xlist = {}
		self.clipping = clipping

	def set_l_rate(self,new_l_rate):
		self.l_rate.set_value(new_l_rate,borrow=True)

	def get_l_rate(self):
		return self.l_rate.get_value(borrow=True)

	def clip(self,matrix):
		if(self.clipping == False):
			return matrix
		else:
			return T.clip(matrix,-1.0,1.0)

	def register_variable(self,variable_name, rows, cols):
		return

	def get_grad_update(self,variable_name,grad_matrix):
		grad_matrix = self.clip(grad_matrix)
		return - (self.l_rate * grad_matrix),[]


class CM(Optimizer):
	'''   Classical momentum method = SGD + momentum      '''

	def __init__(self, l_rate, momentum, clipping):
		Optimizer.__init__(self,l_rate, clipping)
		self.momentum = momentum

	def register_variable(self, variable_name, rows, cols):
		if(rows!=1):
			self.xlist[variable_name] = theano.shared(value=numpy.zeros((rows,cols), dtype=theano.config.floatX), borrow=True)
		else:
			self.xlist[variable_name] = theano.shared(value=numpy.zeros(cols, dtype=theano.config.floatX), borrow=True)

	def get_grad_update(self, variable_name, grad_matrix):
		up = []
		grad_matrix = self.clip(grad_matrix)
		update = self.momentum * self.xlist[variable_name] - self.l_rate * grad_matrix
		up.append((self.xlist[variable_name], update))
		return update,up

class NAG(Optimizer):

	def __init__(self, l_rate, momentum, clipping):
		Optimizer.__init__(self, l_rate, clipping)
		self.momentum = momentum

	def register_variable(self, variable_name, rows, cols):
		if(rows!=1):
			self.xlist[variable_name] = theano.shared(value=numpy.zeros((rows,cols), dtype=theano.config.floatX), borrow=True)
		else:
			self.xlist[variable_name] = theano.shared(value=numpy.zeros(cols, dtype=theano.config.floatX), borrow=True)

	def get_grad_update(self, variable_name, grad_matrix):
		up = []
		grad_matrix = self.clip(grad_matrix)
		x = self.xlist[variable_name]
		new_x = self.momentum * self.xlist[variable_name] + self.l_rate * grad_matrix
		update = self.momentum * x - (1.0 + self.momentum) * new_x
		up.append((self.xlist[variable_name], new_x))
		return update,up


class RMSProp(Optimizer):

	def __init__(self, l_rate, epsilon, decay, clipping):
		Optimizer.__init__(self, l_rate, clipping)
		self.epsilon = epsilon
		self.decay = decay

	def register_variable(self, variable_name, rows, cols):
		if(rows!=1):
			self.wlist[variable_name] = theano.shared(value=numpy.zeros((rows,cols),dtype=theano.config.floatX),borrow=True)
		else:
			self.wlist[variable_name] = theano.shared(value=numpy.zeros(cols,dtype=theano.config.floatX),borrow=True)

	def get_grad_update(self, variable_name, grad_matrix):
		up = []
		grad_matrix = self.clip(grad_matrix)
		new_w = self.decay * self.wlist[variable_name] + (1 - self.decay) * (grad_matrix ** 2)
		rms_w = T.sqrt(new_w+self.epsilon)
		update =  - ((self.l_rate / rms_w) * grad_matrix)
		up.append((self.wlist[variable_name],new_w))
		return update, up


class Adagrad(Optimizer):

	def __init__(self, l_rate, epsilon, clipping):
		Optimizer.__init__(self,l_rate, clipping)
		self.epsilon = epsilon

	def register_variable(self, variable_name, rows, cols):
		if(rows!=1):
			self.wlist[variable_name] = theano.shared(value=numpy.zeros((rows,cols),dtype=theano.config.floatX),borrow=True)
		else:
			self.wlist[variable_name] = theano.shared(value=numpy.zeros(cols,dtype=theano.config.floatX),borrow=True)

	def get_grad_update(self, variable_name, grad_matrix):
		up = []
		grad_matrix = self.clip(grad_matrix)
		new_w = self.wlist[variable_name] + (grad_matrix ** 2)
		rms_w = T.sqrt(new_w+self.epsilon)
		up.append((self.wlist[variable_name], new_w))
		update = - ((self.l_rate/rms_w) * grad_matrix)
		return update,up


class Adadelta(Optimizer):

	def __init__(self,l_rate, epsilon, decay, clipping):
		Optimizer.__init__(self,l_rate, clipping)
		self.epsilon = epsilon
		self.decay = decay

	def register_variable(self, variable_name, rows, cols):
		if(rows!=1):
			self.wlist[variable_name] = theano.shared(value=numpy.zeros((rows,cols),dtype=theano.config.floatX),borrow=True)
			self.xlist[variable_name] = theano.shared(value=numpy.zeros((rows,cols),dtype=theano.config.floatX),borrow=True)
		else:
			self.wlist[variable_name] = theano.shared(value=numpy.zeros(cols,dtype=theano.config.floatX),borrow=True)
			self.xlist[variable_name] = theano.shared(value=numpy.zeros(cols,dtype=theano.config.floatX),borrow=True)

	def get_grad_update(self, variable_name, grad_matrix):
		up = []
		grad_matrix = self.clip(grad_matrix)
		new_w = self.decay * self.wlist[variable_name] + (1 - self.decay) * (grad_matrix ** 2)
		rms_w = T.sqrt(new_w+self.epsilon)
		rms_x_old = T.sqrt(self.xlist[variable_name]+self.epsilon)
		update =  - ((rms_x_old / rms_w) * grad_matrix)
		new_x = self.decay * self.xlist[variable_name] + (1 - self.decay) * (update ** 2)
		up.append((self.xlist[variable_name],new_x))
		up.append((self.wlist[variable_name], new_w))
		return update,up




