__author__ = 'Sarath'

import numpy
import os
import sys

def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)


def get_mat(fname):

	file = open(fname,"r")
	mat = list()
	for line in file:
		line = line.strip().split()
		mat.append(line)
	mat = numpy.asarray(mat,dtype="float32")
	return mat


def get_mat1(folder,mat1,mat2):
	file = open(folder+"ip.txt","w")
	flag = 0

	smat1 = numpy.zeros((1000,len(mat1[0])))
	smat2 = numpy.zeros((1000,len(mat2[0])))
	i = 0
	while(i!=len(mat1)):
		flag = 1
		smat1[i%1000] = mat1[i]
		smat2[i%1000] = mat2[i]
		i+=1
		if(i%1000==0):
			numpy.save(folder+str(i/1000)+"_left",smat1)
			numpy.save(folder+str(i/1000)+"_right",smat2)
			file.write("xy,dense,"+folder+str(i/1000)+",1000\n")
			smat1 = numpy.zeros((1000,len(mat1[0])))
			smat2 = numpy.zeros((1000,len(mat2[0])))
			flag = 0

	if(flag!=0):
		numpy.save(folder+str((i/1000) +1)+"_left",smat1)
		numpy.save(folder+str((i/1000) +1)+"_right",smat2)
		file.write("xy,dense,"+folder+str((i/1000) +1)+","+str(i%1000)+"\n")
	file.close()



def converter(folder):

	create_folder(folder+"matpic/")
	create_folder(folder+"matpic/train")
	create_folder(folder+"matpic/valid")
	create_folder(folder+"matpic/test")

	create_folder(folder+"matpic1/")
	create_folder(folder+"matpic1/train")
	create_folder(folder+"matpic1/valid")
	create_folder(folder+"matpic1/test")


	mat1 = get_mat(folder+"train-view1.txt")
	mat2 = get_mat(folder+"train-view2.txt")
	get_mat1(folder+"matpic1/train/",mat1,mat2)


	mat1 = get_mat(folder+"valid1-view1.txt")
	numpy.save(folder+"matpic/train/view1",mat1)
	mat2 = get_mat(folder+"valid1-view2.txt")
	numpy.save(folder+"matpic/train/view2",mat2)
	numpy.save(folder+"matpic/train/labels",get_mat(folder+"valid1-labels.txt"))

	
	mat1 = get_mat(folder+"valid2-view1.txt")
	numpy.save(folder+"matpic/valid/view1",mat1)
	mat2 = get_mat(folder+"valid2-view2.txt")
	numpy.save(folder+"matpic/valid/view2",mat2)
	get_mat1(folder+"matpic1/valid/",mat1,mat2)
	numpy.save(folder+"matpic/valid/labels",get_mat(folder+"valid2-labels.txt"))

	mat1 = get_mat(folder+"test-view1.txt")
	numpy.save(folder+"matpic/test/view1",mat1)
	mat2 = get_mat(folder+"test-view2.txt")
	numpy.save(folder+"matpic/test/view2",mat2)
	get_mat1(folder+"matpic1/test/",mat1,mat2)
	numpy.save(folder+"matpic/test/labels",get_mat(folder+"test-labels.txt"))



converter(sys.argv[1])