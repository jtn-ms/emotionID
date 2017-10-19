import numpy as np
import neurolab as nl
import config as cfg
import os
import os.path
import math
import util as utl
import cv2

class RNN(object):
    def __init__(self, traindir='./train/', mode='REAL', nettype='HopField'):
        # set parameter1
        self.traindir = traindir
        self.netname = 'rnn'
        self.netfilename = self.netname + '.net'
        self.nettype = nettype
        self.mode = mode
        # set parameter2
        if mode == 'REAL':
            self.feature_size = cfg.FEATURE_SIZE
            self.input_dim = cfg.DIM
            self.input_size = self.input_dim * self.feature_size
            self.cls_cnt = cfg.CLASS_SIZE
        else:
            self.feature_size = 5#np.array(cfg.target).shape[2]
            self.input_dim = 7#np.array(cfg.target).shape[1]
            self.input_size = self.input_dim * self.feature_size
            self.cls_cnt = np.array(cfg.target).shape[0]
        # train or load net
        if os.path.isfile(self.netfilename) and mode == 'REAL':
            self.net = self.loadnet(self.netfilename)
        else:
            self.net = self.train()
            if  mode == 'REAL':
                self.savenet(self.netfilename)

    #set mode
    def setmode(self,mode):
        self.mode = mode
	
    #set type
    def settype(self,nettype):
        self.nettype = nettype

    #set train data
    def setTrainData4FromConfig(self):
        #set in
        x = np.array(cfg.targEt)
        targets = x.reshape(self.cls_cnt,self.input_size)
        return targets

    def setTrainDataFromFile(self):	
        targets = np.empty((self.cls_cnt,self.input_size))
        # load	
        index = 0
        for filename in os.listdir(self.traindir):
            x=np.fromfile(self.traindir+filename,dtype=np.int)
            out = cv2.resize(utl.arr2img(x.reshape((cfg.DIM,cfg.FEATURE_SIZE))),(200,200))
            #cv2.imshow('out' + filename,out)
            #cv2.waitKey(10000)
            target=x.reshape((self.input_size,))
            targets[index,:] = target[:]
            index += 1
        return targets
	
    #train	
    def train(self):
        print('load targets')
        # prepare training data
        if self.mode == 'REAL':
            targets = self.setTrainDataFromFile()
        else:
            targets = self.setTrainData4FromConfig()
        # train
        print('rnn: training starts...')
        if self.nettype == 'HopField':
            return nl.net.newhop(targets)
        else:
            return nl.net.newhem(targets)
        print('rnn: training finished...')
    #test
    def test(self,signal):
        result = self.net.sim(signal)
        #print result\
        return result
    #load
    def loadnet(self,filename):
        print('loading rnn from')#,filename
        return nl.load(filename)
    #save
    def savenet(self,filename):
        print('saving rnn')
        self.net.save(filename)

'''
Reference
https://pythonhosted.org/neurolab/lib.html#module-neurolab.train
https://pythonhosted.org/neurolab/ex_newhop.html
https://pythonhosted.org/neurolab/ex_newhem.html
'''
