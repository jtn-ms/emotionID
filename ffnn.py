import numpy as np
import neurolab as nl
import config as cfg
import os
import os.path
#import math
import util as utl
#import cv2

class FFNN(object):
    def __init__(self, traindir='./train/',mode='REAL'):
        self.traindir = traindir
        self.netname = 'ffnn'
        self.netfilename = self.netname + '.net'
        self.mode = mode
        
        if  mode == 'REAL':
            self.feature_size = cfg.FEATURE_SIZE
            self.input_dim = cfg.DIM
            self.input_size = self.input_dim * self.feature_size
            self.cls_cnt = cfg.CLASS_SIZE
        else:
            self.feature_size = 5#np.array(cfg.target).shape[2]
            self.input_dim = 7#np.array(cfg.target).shape[1]
            self.input_size = self.input_dim * self.feature_size
            self.cls_cnt = np.array(cfg.target).shape[0]

        if  os.path.isfile(self.netfilename) and mode == 'REAL':
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
        inps = utl.ip_2darr(x,self.input_dim)
        #set out
        if  self.cls_cnt == 2:
            y = np.array([-1,1])
        else:
            y = np.linspace(-1.0, 1.0, num=self.cls_cnt)	
        outs = y.reshape(self.cls_cnt,1)
        return inps,outs

    def setTrainDataFromFile(self):
        #set in
        inps = np.empty((self.cls_cnt,self.input_dim))
        index = 0
        for filename in cfg.CLASS_NAME:#os.listdir(self.traindir):
            print self.traindir+filename
            x=np.fromfile(self.traindir+filename+cfg.FILE_EXTENSION,dtype=np.int)
            # if size changed
            #if  len(inp) != self.input_dim:
            #	inp = utl.resizeDim(inp,self.input_dim)
            # endif
            #x=np.reshape(x,(self.input_dim,self.feature_size))
            inp = utl.ip_arr(x,self.input_dim)
            inps[index,:] = inp[:]
            index += 1	
        #set out
        y = np.linspace(-1.0, 1.0, num=self.cls_cnt)
        outs = y.reshape(self.cls_cnt,1)	
        return inps,outs

    #train	
    def initializeNet(self,inps,nnlayout = [20,1]):
        #initialize net	
        MAX = np.amax(inps, axis=0)
        MIN = np.amin(inps, axis=0)
        minmax = np.zeros((self.input_dim,2))
        for i in range(self.input_dim):
            if  MAX[i] == MIN[i]:
                MAX[i] += 1
            minmax[i] = (MIN[i],MAX[i])
        return nl.net.newff(minmax,nnlayout)
        #return nl.net.newff([[0,100]]*self.input_dim,nnlayout)
	
    def train(self):
        # prepare training data
        if  self.mode == 'REAL':
            inps,outs = self.setTrainDataFromFile()
        else:
            inps,outs = self.setTrainData4FromConfig()

        # initialize net
        print 'ffnn: being initialized..'
        net = self.initializeNet(inps)
        # train net
        print 'ffnn: training starts...'
        error = net.train(inps, outs, epochs=500000, show=100, goal=0.00001)
        print 'ffnn: training finished...'
        return net
	
    #test
    def test(self, signal):
        if  len(signal.shape) == 1:
            #sig = np.zeros((1,signal.shape[0]))
            #sig[0,:] = signal[:]
            signal = signal.reshape((1,signal.shape[0]))
        signal = utl.ip_2darr_(signal,self.input_dim)
        result = self.net.sim(signal)
        return result

    #set label
    def getresult(self,signal,filename=''):
        actual = np.linspace(-1.0, 1.0, num=self.cls_cnt)
        predict = self.test(signal)
        dis = np.zeros((predict.shape[0],self.cls_cnt))

        for i in range(predict.shape[0]):
            for j in range(self.cls_cnt):
                dis[i][j] = abs(actual[j] - predict[i])
	
        result = np.zeros((predict.shape[0],1),np.int)

        for i in range(predict.shape[0]):
            for j in range(self.cls_cnt):	
                if  np.min(dis[i]) == dis[i][j]:
                    result[i][0] = j
                    break
        return result

    #load
    def loadnet(self,filename):
        print 'loading ffnn from',filename
        return nl.load(filename)

    #save
    def savenet(self,filename):
        print 'saving ffnn'
        self.net.save(filename)

'''
Reference
https://pythonhosted.org/neurolab/lib.html#module-neurolab.train
https://pythonhosted.org/neurolab/ex_newff.html
'''
