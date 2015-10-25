__author__ = 'Sarath'

import numpy
import os
import sys

sys.path.append("../Model/")
from corrnet import *

def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)



src_folder = sys.argv[1]+"matpic/"
tgt_folder = sys.argv[2]

model = CorrNet()
model.load(tgt_folder)

create_folder(tgt_folder+"project/")

mat = numpy.load(src_folder+"train/view1.npy")
new_mat = model.proj_from_left(mat)
numpy.save(tgt_folder+"project/train-view1",new_mat)

mat = numpy.load(src_folder+"train/view2.npy")
new_mat = model.proj_from_right(mat)
numpy.save(tgt_folder+"project/train-view2",new_mat)

mat = numpy.load(src_folder+"train/labels.npy")
numpy.save(tgt_folder+"project/train-labels",mat)


mat = numpy.load(src_folder+"valid/view1.npy")
new_mat = model.proj_from_left(mat)
numpy.save(tgt_folder+"project/valid-view1",new_mat)

mat = numpy.load(src_folder+"valid/view2.npy")
new_mat = model.proj_from_right(mat)
numpy.save(tgt_folder+"project/valid-view2",new_mat)

mat = numpy.load(src_folder+"valid/labels.npy")
numpy.save(tgt_folder+"project/valid-labels",mat)


mat = numpy.load(src_folder+"test/view1.npy")
new_mat = model.proj_from_left(mat)
numpy.save(tgt_folder+"project/test-view1",new_mat)

mat = numpy.load(src_folder+"test/view2.npy")
new_mat = model.proj_from_right(mat)
numpy.save(tgt_folder+"project/test-view2",new_mat)

mat = numpy.load(src_folder+"test/labels.npy")
numpy.save(tgt_folder+"project/test-labels",mat)

