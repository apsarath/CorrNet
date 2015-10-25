__author__ = 'Sarath'

import os
import numpy
from scipy import sparse

import theano.tensor as T


def create_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

def denseTheanoloader(file, x, bit):
	mat = denseloader(file, bit)
	x.set_value(mat, borrow=True)


def sparseTheanoloader(file, x, bit, row, col):
	mat = sparseloader(file, bit, row, col)
	x.set_value(mat, borrow=True)


def denseloader(file, bit):
	# print "loading ...", file
	matrix = numpy.load(file + ".npy")
	matrix = numpy.array(matrix, dtype=bit)
	return matrix


def sparseloader(file, bit, row, col):
	print "loading ...", file
	x = numpy.load(file + "d.npy")
	y = numpy.load(file + "i.npy")
	z = numpy.load(file + "p.npy")
	matrix = sparse.csr_matrix((x, y, z), shape=(row, col), dtype=bit)
	matrix = matrix.todense()
	return matrix



def activation(x, function):
	if (function == "sigmoid"):
		return T.nnet.sigmoid(x)
	elif (function == "tanh"):
		return T.tanh(x)
	elif (function == "identity"):
		return x
	elif (function == "softmax"):
		return T.nnet.softmax(x)
	elif (function == "softplus"):
		return T.nnet.softplus(x)
	elif (function == "relu"):
		return T.switch(x < 0, 0, x)


def loss(pred, tgt, function):
	if (function == "squarrederror"):
		return T.sum(T.sqr(tgt - pred) / 2, axis=1)
	elif (function == "crossentrophy"):
		return -T.sum(tgt * T.log(pred) + (1 - tgt) * T.log(1 - pred), axis=1)


