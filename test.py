import config as cfg
import rnn as rf
import ffnn as lgl
import util as utl
import numpy as np
import cv2
import os
import neurolab as nl

'''
n1 = rf.RNN('','NotReal')
n2 = lgl.FFNN('','NotReal')
signal = np.array(cfg.inpuT2)
print signal
refined = n1.test(signal)
print refined
final = n2.getresult(refined)
print final
signal = signal.reshape((signal.shape[0],7,5))
refined = refined.reshape((signal.shape[0],7,5))
inp = cv2.resize(utl.arr2img(signal[0]),(100,100))
out = cv2.resize(utl.arr2img(refined[0]),(100,100))
cv2.imshow('in.png',inp)
cv2.imshow('out.png',out)
cv2.waitKey(10000)
'''
#print utl.int2bytes(10,10)

#img-->
def testReal():
    tgt_path = './train/'
    test_cls_name = 'T'
    trandir = tgt_path + test_cls_name + '/'
    testdir = tgt_path + test_cls_name + '/'

    n1 = rf.RNN(trandir,'REAL')
    n2 = lgl.FFNN(trandir)

    filecount = 0
    for filename in os.listdir(testdir):
    	filecount += 1

    signals = np.zeros((filecount,cfg.INPUT_SIZE),np.int)
    index = 0
    for filename in os.listdir(testdir):
    	signal = np.fromfile(testdir + filename, dtype=np.int)
    	#if  len(signal) != cfg.INPUT_SIZE:
    	#    signal = utl.resizeDim(signal,cfg.DIM)
    	signal = signal.reshape((cfg.INPUT_SIZE,))
    	signals[index,:] = signal[:]
    	index += 1

    refined = n1.test(signals)
    for i in range(filecount):
    	inp = cv2.resize(utl.arr2img(signals[i].reshape((cfg.DIM,cfg.FEATURE_SIZE))),(200,200))
    	out = cv2.resize(utl.arr2img(refined[i].reshape((cfg.DIM,cfg.FEATURE_SIZE))),(200,200))
    	cv2.imshow('in' + str(i),inp)
    	cv2.imshow('out' + str(i),out)
    cv2.waitKey(100000)
    final = n2.getresult(refined)
    print final

def checkSamples(testdir):
    filecount = 0
    for filename in os.listdir(testdir):
    	filecount += 1

    signals = np.zeros((filecount,cfg.INPUT_SIZE),np.int)
    index = 0
    for filename in os.listdir(testdir):
    	signal = np.fromfile(testdir + filename, dtype=np.int)
    	#if  len(signal) != cfg.INPUT_SIZE:
    	#    signal = utl.resizeDim(signal,cfg.DIM)
    	signal = signal.reshape((cfg.INPUT_SIZE,))
    	signals[index,:] = signal[:]
    	index += 1
    #for i in range(filecount):
    	inp = cv2.resize(utl.arr2img(signal.reshape((cfg.DIM,cfg.FEATURE_SIZE))),(200,200))
    	cv2.imshow(filename,inp)
    cv2.waitKey(100000)      
  
def testHop():
    tgt_path = './train/'
    test_cls_name = 'T'
    trandir = tgt_path + test_cls_name + '/'
    testdir = tgt_path + test_cls_name + '/'

    filecount = 0
    for filename in os.listdir(trandir):
        filecount += 1

    signals = np.zeros((filecount,cfg.INPUT_SIZE),np.int)
    index = 0
    for filename in os.listdir(trandir):
        signal = np.fromfile(testdir + filename, dtype=np.int)
        #if  len(signal) != cfg.INPUT_SIZE:
        #    signal = utl.resizeDim(signal,cfg.DIM)
        signal = signal.reshape((cfg.INPUT_SIZE,))
        #signals[index,:] = signal[:]
        index += 1
    signals = np.array(cfg.tarGET)
    net = nl.net.newhop(signals)   
    refined = net.sim(signals)
    print signals
    print refined
    for i in range(signals.shape[0]):
        inp = cv2.resize(utl.arr2img(signals[i].reshape((7,20))),(200,200))
        out = cv2.resize(utl.arr2img(refined[i].reshape((7,20))),(200,200))
        cv2.imshow('in' + str(i),inp)
        cv2.imshow('out' + str(i),out)
    cv2.waitKey(1000000)

def testHemming():

    target = [[-1, 1, -1, -1, 1, -1, -1, 1, -1],
          [1, 1, 1, 1, -1, 1, 1, -1, 1],
          [1, -1, 1, 1, 1, 1, 1, -1, 1],
          [1, 1, 1, 1, -1, -1, 1, -1, -1],
          [-1, -1, -1, -1, 1, -1, -1, -1, -1]]

    input = [[-1, -1, 1, 1, 1, 1, 1, -1, 1],
         [-1, -1, 1, -1, 1, -1, -1, -1, -1],
         [-1, -1, -1, -1, 1, -1, -1, 1, -1]]
    
    '''
    target=[[1,0,0,0,1,
             1,1,0,0,1,
             1,0,1,0,1,
             1,0,0,1,1,
             1,0,0,0,1],
            [1,1,1,1,1,
             1,0,0,0,0,
             1,1,1,1,1,
             1,0,0,0,0,
             1,1,1,1,1],
            [1,1,1,1,0,
             1,0,0,0,1,
             1,1,1,1,0,
             1,0,0,1,0,
             1,0,0,0,1],
            [0,1,1,1,0,
             1,0,0,0,1,
             1,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,0]]
    input = [[0,0,0,0,0,
             1,1,0,0,1,
             1,1,0,0,1,
             1,0,1,1,1,
             0,0,0,1,1]]
    '''
    # Create and train network  
    net = nl.net.newhem(target)

    output = net.sim(target)
    print("Test on train samples (must be [0, 1, 2, 3, 4])")
    print(np.argmax(output, axis=0))

    output = net.sim([input[0]])
    print("Outputs on recurent cycle:")
    print(np.array(net.layers[1].outs))

    output = net.sim(input)
    print("Outputs on test sample:")
    print(output)

def testFFNN(trandir,testdir):
    net = lgl.FFNN(trandir)

    filecount = 0
    for filename in os.listdir(testdir):
        filecount += 1

    signals = np.zeros((filecount,cfg.INPUT_SIZE),np.int)
    index = 0
    for filename in os.listdir(testdir):
        print filename
        signal = np.fromfile(testdir + filename, dtype=np.int)       
        signal = signal.reshape((cfg.INPUT_SIZE,))
        signals[index,:] = signal[:]
        img = cv2.resize(utl.arr2img(signals[index].reshape((cfg.DIM,cfg.FEATURE_SIZE))),(200,200))
        cv2.imshow(filename,img)
        index += 1
    #print signals
    result = net.getresult(signals)
    print result
    for i in range(result.shape[0]):
        print(cfg.CLASS_NAME[result[i][0]])        

    cv2.waitKey(100000)  
    
utl.makeSample()
testFFNN('./train/T/','./train/Neutral/')
#checkSamples('./train/T/')
#testHop()
#testHemming()

