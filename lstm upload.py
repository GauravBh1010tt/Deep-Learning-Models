# -*- coding: utf-8 -*-
import sys
#import time
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

theano.config.compute_test_value = 'off'

class lstm:
    
    def __init__(self , v_dim=2000 , h_dim = 100, bptt_trunc = 4):
        
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.bptt_trunc = bptt_trunc
        
        E   = rn.uniform(-np.sqrt(1./v_dim),np.sqrt(1./v_dim),(h_dim,v_dim))
        W_x = rn.uniform(-np.sqrt(1./h_dim),np.sqrt(1./h_dim),(4,h_dim,h_dim))
        H_x = rn.uniform(-np.sqrt(1./h_dim),np.sqrt(1./h_dim),(4,h_dim,h_dim))
        V   = rn.uniform(-np.sqrt(1./h_dim),np.sqrt(1./h_dim),(v_dim,h_dim))
        b   = np.zeros((4,h_dim))                 
        b_o = np.zeros(v_dim)
        
        self.E   = theano.shared(name='E',value=E.astype(theano.config.floatX))
        self.W_x = theano.shared(name='W_x',value=W_x.astype(theano.config.floatX))
        self.H_x = theano.shared(name='H_x',value=H_x.astype(theano.config.floatX))
        self.V   = theano.shared(name='V',value=V.astype(theano.config.floatX))
        self.b   = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.b_o = theano.shared(name='b_o', value=b_o.astype(theano.config.floatX))
        
        self.mE   = theano.shared(name = 'mE', value=np.zeros(E.shape).astype(theano.config.floatX))        
        self.mW_x = theano.shared(name = 'mW_x', value=np.zeros(W_x.shape).astype(theano.config.floatX))
        self.mH_x = theano.shared(name = 'mH_x', value=np.zeros(H_x.shape).astype(theano.config.floatX))
        self.mV   = theano.shared(name = 'mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mb   = theano.shared(name = 'mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mb_o = theano.shared(name = 'mb_o', value=np.zeros(b_o.shape).astype(theano.config.floatX))
        
        self.theano = {}
        self.build_model()

    def build_model(self):
        
        E ,W_x ,H_x ,V ,b , b_o = self.E ,self.W_x ,self.H_x ,self.V ,self.b ,self.b_o
        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop(x_t,s_t,c_t):
            
            #print 'forward'
            x_e = E[:,x_t]
            i_t1 = T.nnet.sigmoid(T.dot(x_e,W_x[0]) + T.dot(s_t,H_x[0]) + b[0])
            f_t1 = T.nnet.sigmoid(T.dot(x_e,W_x[1]) + T.dot(s_t,H_x[1]) + b[1])
            o_t1 = T.nnet.sigmoid(T.dot(x_e,W_x[2]) + T.dot(s_t,H_x[2]) + b[2])
            g_t1 = T.tanh(T.dot(x_e,W_x[3]) + T.dot(s_t,H_x[3]) + b[3])
            c_t1 = (f_t1 * c_t) + (g_t1 * i_t1)
            
            s_t1 = T.tanh(c_t1) * o_t1
            y_t1 = T.nnet.softmax(T.dot(V,s_t1) + b_o)[0]
            #y_t1 = T.nnet.softmax(T.dot(x_e,V) + b_o)[0]
            #print 'ending forward'
            #return [y_t1,s_t1,c_t1]
            #theano.printing.Print('value of')(y_t1)
            return [s_t1,c_t1,y_t1]
        
        [s,c,a] ,update = theano.scan(forward_prop, sequences = x, truncate_gradient = self.bptt_trunc, 
                                    outputs_info=[dict(initial=T.zeros(self.h_dim)),
                                                  dict(initial=T.zeros(self.h_dim)),None])
        pred = T.argmax(a,axis=1)
        cost = T.sum(T.nnet.categorical_crossentropy(a,y))
        
        #self.forward = theano.function([x],a)
       
        gE   = T.grad(cost,E)
        gW_x = T.grad(cost,W_x)
        gH_x = T.grad(cost,H_x)
        gV   = T.grad(cost,V)
        gb   = T.grad(cost,b)
        gb_o = T.grad(cost,b_o)
        
        self.forward = theano.function([x],s)
        self.predict = theano.function([x],pred)
        self.error   = theano.function([x,y],cost)
        self.bptt    = theano.function([x,y],[gE,gW_x,gH_x,gV,gb,gb_o])
        
        alpha = T.scalar('alpha')
        decay = T.scalar('decay')
        
        mE   = decay * self.mE + (1-decay) * gE **2        
        mW_x = decay * self.mW_x + (1-decay) * gW_x **2
        mH_x = decay * self.mH_x + (1-decay) * gH_x **2
        mV   = decay * self.mV + (1-decay) * gV **2
        mb   = decay * self.mb + (1-decay) * gb **2
        mb_o = decay * self.mb_o + (1-decay) * gb_o **2                
        
        #print 'starting train'
        self.train = theano.function([x,y,alpha,decay],[],
                                      updates = [(E, E - alpha * gE / T.sqrt(mE+1e-6)),
                                                 (W_x, W_x - alpha * gW_x / T.sqrt(mW_x+1e-6)),
                                                 (H_x, H_x - alpha * gH_x / T.sqrt(mH_x+1e-6)),
                                                 (V, V - alpha * gV / T.sqrt(mV+1e-6)),
                                                 (b, b - alpha * gb / T.sqrt(mb+1e-6)),
                                                 (b_o, b_o - alpha * gb_o / T.sqrt(mb_o+1e-6)),
                    (self.mE,mE),(self.mW_x,mW_x),(self.mH_x,mH_x),(self.mV,mV),(self.mb,mb),(self.mb_o,mb_o)])
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
    

'''
model = lstm(vocab_size,h_dim = H_dim)
t1 = time.time()
model.train(X_train[0],Y_train[0],Alpha,0.9)
t2 = time.time()

print 'One sample training time :: ',(t2-t1),' sec'

t1 = time.time()
train_with_sgd(model,X_train,Y_train,nepoch=20,learning_rate=Alpha)
t2 = time.time()

print '100 epoch training time :: ',(t2-t1),' sec'''
