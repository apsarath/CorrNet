__author__ = 'Sarath'

import time
import pickle

from optimization import *
from Initializer import *
from NNUtil import *


import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class CorrNet(object):

    def init(self, numpy_rng, theano_rng=None, l_rate=0.01, optimization="sgd",
             tied=False, n_visible_left=None, n_visible_right=None, n_hidden=None, lamda=5,
             W_left=None, W_right=None, b=None, W_left_prime=None, W_right_prime=None,
             b_prime_left=None, b_prime_right=None, input_left=None, input_right=None,
             hidden_activation="sigmoid", output_activation="sigmoid", loss_fn = "squarrederror",
             op_folder=None):

        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.optimization = optimization
        self.l_rate = l_rate

        self.optimizer = get_optimizer(self.optimization, self.l_rate)
        self.Initializer = Initializer(self.numpy_rng)

        self.n_visible_left = n_visible_left
        self.n_visible_right = n_visible_right
        self.n_hidden = n_hidden
        self.lamda = lamda
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.tied = tied
        self.op_folder = op_folder

        self.W_left = self.Initializer.fan_based_sigmoid("W_left", W_left, n_visible_left, n_hidden)
        self.optimizer.register_variable("W_left",n_visible_left,n_hidden)

        self.W_right = self.Initializer.fan_based_sigmoid("W_right", W_right, n_visible_right, n_hidden)
        self.optimizer.register_variable("W_right",n_visible_right,n_hidden)

        if not tied:
            self.W_left_prime = self.Initializer.fan_based_sigmoid("W_left_prime", W_left_prime, n_hidden, n_visible_left)
            self.optimizer.register_variable("W_left_prime",n_hidden, n_visible_left)
            self.W_right_prime = self.Initializer.fan_based_sigmoid("W_right_prime", W_right_prime, n_hidden, n_visible_right)
            self.optimizer.register_variable("W_right_prime",n_hidden, n_visible_right)
        else:
            self.W_left_prime = self.W_left.T
            self.W_right_prime = self.W_right.T

        self.b = self.Initializer.zero_vector("b", b, n_hidden)
        self.optimizer.register_variable("b",1,n_hidden)

        self.b_prime_left = self.Initializer.zero_vector("b_prime_left", b_prime_left, n_visible_left)
        self.optimizer.register_variable("b_prime_left",1,n_visible_left)
        self.b_prime_right = self.Initializer.zero_vector("b_prime_right", b_prime_right, n_visible_right)
        self.optimizer.register_variable("b_prime_right",1,n_visible_right)

        if input_left is None:
            self.x_left = T.matrix(name='x_left')
        else:
            self.x_left = input_left

        if input_right is None:
            self.x_right = T.matrix(name='x_right')
        else:
            self.x_right = input_right


        if tied:
            self.params = [self.W_left, self.W_right,  self.b, self.b_prime_left, self.b_prime_right]
            self.param_names = ["W_left", "W_right", "b", "b_prime_left", "b_prime_right"]
        else:
            self.params = [self.W_left, self.W_right,  self.b, self.b_prime_left, self.b_prime_right, self.W_left_prime, self.W_right_prime]
            self.param_names = ["W_left", "W_right", "b", "b_prime_left", "b_prime_right", "W_left_prime", "W_right_prime"]


        self.proj_from_left = theano.function([self.x_left],self.project_from_left())
        self.proj_from_right = theano.function([self.x_right],self.project_from_right())
        self.recon_from_left = theano.function([self.x_left],self.reconstruct_from_left())
        self.recon_from_right = theano.function([self.x_right],self.reconstruct_from_right())

        self.save_params()


    def train_common(self,mtype="1111"):

        y1_pre = T.dot(self.x_left, self.W_left) + self.b
        y1 = activation(y1_pre, self.hidden_activation)
        z1_left_pre = T.dot(y1, self.W_left_prime) + self.b_prime_left
        z1_right_pre = T.dot(y1,self.W_right_prime) + self.b_prime_right
        z1_left = activation(z1_left_pre, self.output_activation)
        z1_right = activation(z1_right_pre, self.output_activation)
        L1 = loss(z1_left, self.x_left, self.loss_fn) + loss(z1_right, self.x_right, self.loss_fn)

        y2_pre = T.dot(self.x_right, self.W_right) + self.b
        y2 = activation(y2_pre, self.hidden_activation)
        z2_left_pre = T.dot(y2, self.W_left_prime) + self.b_prime_left
        z2_right_pre = T.dot(y2,self.W_right_prime) + self.b_prime_right
        z2_left = activation(z2_left_pre, self.output_activation)
        z2_right = activation(z2_right_pre, self.output_activation)
        L2 = loss(z2_left, self.x_left, self.loss_fn) + loss(z2_right, self.x_right, self.loss_fn)

        y3_pre = T.dot(self.x_left, self.W_left) + T.dot(self.x_right, self.W_right) + self.b
        y3 = activation(y3_pre, self.hidden_activation)
        z3_left_pre = T.dot(y3, self.W_left_prime) + self.b_prime_left
        z3_right_pre = T.dot(y3,self.W_right_prime) + self.b_prime_right
        z3_left = activation(z3_left_pre, self.output_activation)
        z3_right = activation(z3_right_pre, self.output_activation)
        L3 = loss(z3_left, self.x_left, self.loss_fn) + loss(z3_right, self.x_right, self.loss_fn)

        y1_mean = T.mean(y1, axis=0)
        y1_centered = y1 - y1_mean
        y2_mean = T.mean(y2, axis=0)
        y2_centered = y2 - y2_mean
        corr_nr = T.sum(y1_centered * y2_centered, axis=0)
        corr_dr1 = T.sqrt(T.sum(y1_centered * y1_centered, axis=0)+1e-8)
        corr_dr2 = T.sqrt(T.sum(y2_centered * y2_centered, axis=0)+1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr/corr_dr
        L4 = T.sum(corr) * self.lamda

        ly4_pre = T.dot(self.x_left, self.W_left) + self.b
        ly4 = activation(ly4_pre, self.hidden_activation)
        lz4_right_pre = T.dot(ly4,self.W_right_prime) + self.b_prime_right
        lz4_right = activation(lz4_right_pre, self.output_activation)
        ry4_pre = T.dot(self.x_right, self.W_right) + self.b
        ry4 = activation(ry4_pre, self.hidden_activation)
        rz4_left_pre = T.dot(ry4,self.W_left_prime) + self.b_prime_left
        rz4_left = activation(rz4_left_pre, self.output_activation)
        L5 = loss(lz4_right, self.x_right, self.loss_fn) + loss(rz4_left, self.x_left, self.loss_fn)

        if mtype=="1111":
            print "1111"
            L = L1 + L2 + L3 - L4
        elif mtype=="1110":
            print "1110"
            L = L1 + L2 + L3
        elif mtype=="1101":
            print "1101"
            L = L1 + L2 - L4
        elif mtype == "0011":
            print "0011"
            L = L3 - L4
        elif mtype == "1100":
            print "1100"
            L = L1 + L2
        elif mtype == "0010":
            print "0010"
            L = L3
        elif mtype == "euc":
            print "euc"
            L = L5
        elif mtype == "euc-cor":
            print "euc-cor"
            L = L5 - L4

        cost = T.mean(L)

        gradients = T.grad(cost, self.params)
        updates = []
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        return cost, updates

    def train_left(self):

        y_pre = T.dot(self.x_left, self.W_left) + self.b
        y = activation(y_pre, self.hidden_activation)
        z_left_pre = T.dot(y, self.W_left_prime) + self.b_prime_left
        z_left = activation(z_left_pre, self.output_activation)
        L = loss(z_left, self.x_left, self.loss_fn)
        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_left, self.b, self.b_prime_left]
            curr_param_names = ["W_left", "b", "b_prime_left"]
        else:
            curr_params = [self.W_left, self.b, self.b_prime_left, self.W_left_prime]
            curr_param_names = ["W_left", "b", "b_prime_left", "W_left_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p,g,n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return cost, updates

    def train_right(self):

        y_pre = T.dot(self.x_right, self.W_right) + self.b
        y = activation(y_pre, self.hidden_activation)
        z_right_pre = T.dot(y, self.W_right_prime) + self.b_prime_right
        z_right = activation(z_right_pre, self.output_activation)
        L = loss(z_right, self.x_right, self.loss_fn)
        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_right, self.b, self.b_prime_right]
            curr_param_names = ["W_right", "b", "b_prime_right"]
        else:
            curr_params = [self.W_right, self.b, self.b_prime_right, self.W_right_prime]
            curr_param_names = ["W_right", "b", "b_prime_right", "W_right_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p,g,n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return cost, updates

    def project_from_left(self):

        y_pre = T.dot(self.x_left, self.W_left) + self.b
        y = activation(y_pre, self.hidden_activation)
        return y

    def project_from_right(self):

        y_pre = T.dot(self.x_right, self.W_right) + self.b
        y = activation(y_pre, self.hidden_activation)
        return y

    def reconstruct_from_left(self):

        y_pre = T.dot(self.x_left, self.W_left) + self.b
        y = activation(y_pre, self.hidden_activation)
        z_left_pre = T.dot(y, self.W_left_prime) + self.b_prime_left
        z_right_pre = T.dot(y,self.W_right_prime) + self.b_prime_right
        z_left = activation(z_left_pre, self.output_activation)
        z_right = activation(z_right_pre, self.output_activation)
        return z_left, z_right

    def reconstruct_from_right(self):

        y_pre = T.dot(self.x_right, self.W_right) + self.b
        y = activation(y_pre, self.hidden_activation)
        z_left_pre = T.dot(y, self.W_left_prime) + self.b_prime_left
        z_right_pre = T.dot(y,self.W_right_prime) + self.b_prime_right
        z_left = activation(z_left_pre, self.output_activation)
        z_right = activation(z_right_pre, self.output_activation)
        return z_left, z_right

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):

        for p,nm in zip(self.params, self.param_names):
            numpy.save(self.op_folder+nm, p.get_value(borrow=True))

    def save_params(self):

        params = {}
        params["optimization"] = self.optimization
        params["l_rate"] = self.l_rate
        params["n_visible_left"] = self.n_visible_left
        params["n_visible_right"] = self.n_visible_right
        params["n_hidden"] = self.n_hidden
        params["lamda"] = self.lamda
        params["hidden_activation"] = self.hidden_activation
        params["output_activation"] = self.output_activation
        params["loss_fn"] = self.loss_fn
        params["tied"] = self.tied
        params["numpy_rng"] = self.numpy_rng
        params["theano_rng"] = self.theano_rng

        pickle.dump(params,open(self.op_folder+"params.pck","wb"),-1)


    def load(self, folder, input_left=None, input_right=None):

        plist = pickle.load(open(folder+"params.pck","rb"))

        self.init(plist["numpy_rng"], theano_rng=plist["theano_rng"], l_rate=plist["l_rate"],
                  optimization=plist["optimization"], tied=plist["tied"],
                  n_visible_left=plist["n_visible_left"], n_visible_right=plist["n_visible_right"],
                  n_hidden=plist["n_hidden"], lamda=plist["lamda"], W_left=folder+"W_left",
                  W_right=folder+"W_right", b=folder+"b", W_left_prime=folder+"W_left_prime",
                  W_right_prime=folder+"W_right_prime", b_prime_left=folder+"b_prime_left",
                  b_prime_right=folder+"b_prime_right", input_left=input_left, input_right=input_right,
                  hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
                  loss_fn = plist["loss_fn"], op_folder=folder)




def trainCorrNet(src_folder, tgt_folder, batch_size = 20, training_epochs=40,
                 l_rate=0.01, optimization="sgd", tied=False, n_visible_left=None,
                 n_visible_right=None, n_hidden=None, lamda=5,
                 W_left=None, W_right=None, b=None, W_left_prime=None, W_right_prime=None,
                 b_prime_left=None, b_prime_right=None, hidden_activation="sigmoid",
                 output_activation="sigmoid", loss_fn = "squarrederror"):

    index = T.lscalar()
    x_left = T.matrix('x_left')
    x_right = T.matrix('x_right')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model = CorrNet()
    model.init(numpy_rng=rng, theano_rng=theano_rng, l_rate=l_rate, optimization=optimization, tied=tied, n_visible_left=n_visible_left, n_visible_right=n_visible_right, n_hidden=n_hidden, lamda=lamda, W_left=W_left, W_right=W_right, b=b, W_left_prime=W_left_prime, W_right_prime=W_right_prime, b_prime_left=b_prime_left, b_prime_right=b_prime_right, input_left=x_left, input_right=x_right, hidden_activation=hidden_activation, output_activation=output_activation, loss_fn =loss_fn, op_folder=tgt_folder)
    #model.load(tgt_folder,x_left,x_right)
    start_time = time.clock()
    train_set_x_left = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_left)), dtype=theano.config.floatX), borrow=True)
    train_set_x_right = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_right)), dtype=theano.config.floatX), borrow=True)

    common_cost, common_updates = model.train_common("1111")
    mtrain_common = theano.function([index], common_cost,updates=common_updates,givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size]),(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])

    left_cost, left_updates = model.train_left()
    mtrain_left = theano.function([index], left_cost,updates=left_updates,givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size])])

    right_cost, right_updates = model.train_right()
    mtrain_right = theano.function([index], right_cost,updates=right_updates,givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])


    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    oldtc = float("inf")

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch
        c = []
        ipfile = open(src_folder+"train/ip.txt","r")
        for line in ipfile:
            next = line.strip().split(",")
            if(next[0]=="xy"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                    denseTheanoloader(next[2]+"_right",train_set_x_right, "float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                    sparseTheanoloader(next[2]+"_right",train_set_x_right, "float32", 1000, n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_common(batch_index))
            elif(next[0]=="x"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_left(batch_index))
            elif(next[0]=="y"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_right",train_set_x_right,"float32")
                else:
                    sparseTheanoloader(next[2]+"_right",train_set_x_right,"float32",1000,n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_right(batch_index))


        if(flag==1):
            flag = 0
            diff = numpy.mean(c)
            di = diff
        else:
            di = numpy.mean(c) - diff
            diff = numpy.mean(c)

        print 'Difference between 2 epochs is ', di
        print 'Training epoch %d, cost ' % epoch, diff

        ipfile.close()

        detfile = open(tgt_folder+"details.txt","a")
        detfile.write("train\t"+str(diff)+"\n")
        detfile.close()
        # save the parameters for every 5 epochs
        if((epoch+1)%5==0):
            model.save_matrices()

    end_time = time.clock()
    training_time = (end_time - start_time)
    print ' code ran for %.2fm' % (training_time / 60.)
    model.save_matrices()
