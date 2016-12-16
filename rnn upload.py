# -*- coding: utf-8 -*-
import sys
import os
import time
import nltk
import csv
import pickle
import sklearn
import theano
import itertools
import numpy as np
import theano.tensor as T
import numpy.random as rn
from datetime import datetime
from nn.math import softmax, make_onehot
from nltk.tokenize import regexp_tokenize

#theano.config.floatX = 'float32'
#theano.config.device = 'gpu'

class rnn_theano:
    
    def __init__(self, v_dim=2000, h_dim=100, bptt_trunc = 4, saved_model=None):
        
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.bptt_trunc = bptt_trunc
        
        if saved_model == None:
            L = rn.uniform(-np.sqrt(1./v_dim),np.sqrt(1./v_dim),(h_dim,v_dim))
            U = rn.uniform(-np.sqrt(1./h_dim),np.sqrt(1./h_dim),(v_dim,h_dim))
            H = rn.uniform(-np.sqrt(1./h_dim),np.sqrt(1./h_dim),(h_dim,h_dim))
        else:
            saved_model = load_model()
            L = saved_model[0]
            U = saved_model[1]
            H = saved_model[2]
        
        self.L = theano.shared(name='L',value=L.astype(theano.config.floatX))
        self.U = theano.shared(name='U',value=U.astype(theano.config.floatX))
        self.H = theano.shared(name='H',value=H.astype(theano.config.floatX))
        
        self.theano = {}
        self.build_model()
    
    def build_model(self):
        
        L, U, H = self.L, self.U, self.H
        x = T.ivector('x')
        y = T.ivector('y')
        #h = T.ivector('h')
        
        def forward_prop(x_t, h_t, L, U, H):
            h = T.nnet.sigmoid(T.dot(H,h_t)+L[:,x_t])
            #print 'h_t',h_t
            y_h = T.nnet.softmax(T.dot(U,h))
            return [y_h[0],h]
        [a,s] , update = theano.scan(forward_prop, sequences=x,
                                 outputs_info = [None, dict(initial=T.zeros(self.h_dim))],                               non_sequences=[L, U, H],truncate_gradient=self.bptt_trunc)      
        pred = T.argmax(a,axis=1)
        cost = T.sum(T.nnet.categorical_crossentropy(a,y))
        gU = T.grad(cost,U)
        gL = T.grad(cost,L)                
        gH = T.grad(cost,H)
        
        self.forward = theano.function([x],a)
        self.predict = theano.function([x],pred)
        self.error = theano.function([x,y],cost)
        self.bptt = theano.function([x,y],[gU,gL,gH])
        
        alpha = T.scalar('alpha')
        self.train = theano.function([x,y,alpha],[],updates=((self.U,self.U-alpha*gU),
                                (self.L,self.L-alpha*gL),(self.H,self.H-alpha*gH)))
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   

        
def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    model.bptt_truncate = 1000
    bptt_gradients = model.bptt(x, y)
    model_parameters = ['L', 'U', 'H']
    for pidx, pname in enumerate(model_parameters):
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            original_value = parameter[ix]
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            backprop_gradient = bptt_gradients[pidx][ix]
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient:" ,backprop_gradient
                print "Relative Error: %f" % relative_error
                return 
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)
    
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1):
    print '......started training with SGD......' 
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        for i in range(len(y_train)):
            # One SGD step
            if num_examples_seen %10 == 0:
                print num_examples_seen
            if (num_examples_seen % 100 == 0):
                #print 'inside'
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)

            model.train(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def prepare_data(sent,vocab_size=2000):
    
    START = '$_START_$'
    END = '$_END_$'
    unk_token = '$_UNK_$'  
    sentence = ["%s %s %s" % (START,x,END) for x in sent]
    tokenize_sent = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
    
    freq = nltk.FreqDist(itertools.chain(*tokenize_sent))
    print 'found ',len(freq),' unique words'
    vocab = freq.most_common(vocab_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unk_token)
    
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    for i,sent in enumerate(tokenize_sent):
        tokenize_sent[i] = [w if w in word_to_index else unk_token for w in sent]
    #X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenize_sent])
    X_train = []
    for i in tokenize_sent:
        temp = []
        for j in i[:-1]:
            temp.append(word_to_index[j])
        X_train.append(temp)    
    X_train = np.asarray(X_train)
    Y_train = []
    for i in tokenize_sent:
        temp = []
        for j in i[1:]:
            temp.append(word_to_index[j])
        Y_train.append(temp)
    Y_train = np.asarray(Y_train)
    
    return X_train,Y_train

def load_model():
    saved_model = []
    with open('model_l') as h:
        a=pickle.loads(h.read())
    saved_model.append(a)
    
    with open('model_u') as h:
        a=pickle.loads(h.read())
    saved_model.append(a)
    
    with open('model_h') as h:
        a=pickle.loads(h.read())
    saved_model.append(a)
    
    return saved_model

#def save_model(model):
    
#model = rnn_theano(vocab_size,h_dim = H_dim,saved_model=saved_model)
#t1 = time.time()
#model.train(X_train[0],Y_train[0],Alpha)
#t2 = time.time()

#print 'One sample training time :: ',(t2-t1),' sec'

#t1 = time.time()
#train_with_sgd(model,X_train[:100],Y_train[:100],nepoch=3,learning_rate=Alpha)
#t2 = time.time()

#print '100 epoch training time :: ',(t2-t1),' sec'
