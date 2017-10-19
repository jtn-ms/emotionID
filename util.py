import numpy as np
#import random
#import Image
import os
#import re
import cv2
import config as cfg
import math

#Matrix to Vector
def arr2vec(arr,filename = None):
    size = 1
    for i in range(len(arr.shape)):
        size *= arr.shape[i]
    vec = arr.reshape((size,))
    if  filename is not None:
        arr.tofile(filename + '.vec')	
    return vec

#draw line
def drawLine(img,shape,here,there,isClosed=False,linewidth=2):
    for i in range(here,there):
        cv2.line(img,(shape[i][0],shape[i][1]),(shape[i + 1][0],shape[i + 1][1]),255,linewidth)		
    if  isClosed:
        cv2.line(img,(shape[here][0],shape[here][1]),(shape[there][0],shape[there][1]),255,linewidth)

#draw contour
def drawContours(img,coords,linewidth=2):
    drawLine(img,coords,0,16)
    drawLine(img,coords,17,21)
    drawLine(img,coords,22,26)
    drawLine(img,coords,27,30)
    drawLine(img,coords,30,35,True)
    drawLine(img,coords,36,41,True)
    drawLine(img,coords,42,47,True)
    drawLine(img,coords,48,59,True)
    drawLine(img,coords,60,67,True)
    cv2.line(img,(coords[48][0],coords[48][1]),(coords[60][0],coords[60][1]),255,linewidth)
    cv2.line(img,(coords[54][0],coords[54][1]),(coords[64][0],coords[64][1]),255,linewidth)

#Array to Image
def arr2img(arr, filename = None):
    #data is 1 or -1 matrix
    if  len(arr.shape) == 1:
        dim = int(math.sqrt(len(arr)))
        arr = arr.reshape((dim,dim))
    img = np.zeros(arr.shape, np.uint8)
    img[arr==1] = 255
    img[arr==-1] = 0
    if filename is not None:
        cv2.imwrite(filename + '.png',img)
    return img

#Image to Array
def img2arr(img, filename = None):
    arr = np.ones((img.shape),np.int)
    arr[:] = -1
    arr[img > 128] = 1
    if  filename is not None:
        fullname = filename + '.dat'
        arr.tofile(fullname)
    return arr

#facial landmarks to feature space(array)
def fpoints2array(points, filename = None, newdim = cfg.DIM):
    # INT2FLOAT
    points = points.astype(float)
    #
    MAX = np.amax(points, axis=0)
    MIN = np.amin(points, axis=0)
    width = MAX.item(0) - MIN.item(0)
    height = MAX.item(1) - MIN.item(1)
    # MOVE
    points -= [MIN[0],MIN[1]]
    # RESIZE
    x_ratio = newdim / width
    y_ratio = newdim / height
    points *= [x_ratio,y_ratio]
    # FLOAT2INT
    points = np.ceil(points)
    points = points.astype(int)
    points[points == newdim] = newdim - 1
    # POINT2IMG
    img = np.zeros((newdim, newdim), np.uint8)
    drawContours(img,points) 
    if  filename is not None:
        cv2.imwrite(filename + '.png',img)
        cv2.imshow('test',img)
        cv2.waitKey(1)
    # IMG2ARR
    arr = img2arr(img,filename) 

    return img,arr

# pts to arr
def pts2arr(points,newdim):
    # INT2FLOAT
    points = points.astype(float)
    #
    MAX = np.amax(points, axis=0)
    MIN = np.amin(points, axis=0)
    width = MAX.item(0) - MIN.item(0)
    height = MAX.item(1) - MIN.item(1)
    # RESIZE
    x_ratio = newdim / width
    y_ratio = newdim / height
    points *= [x_ratio,y_ratio]
    # FLOAT2INT
    points = np.ceil(points)
    points = points.astype(int)
    points[points == newdim] = newdim - 1
    # POINT2IMG
    arr = np.ones((newdim,newdim),np.int)
    arr[:] = -1
    for (x,y) in points:
        arr[x][y] = 1
    	
    return arr
#dis betwen point1 and point2
def distance(pt1,pt2):
    return (pt1[0]-pt2[0])**2 + (pt1[1] - pt2[1])**2

# int to bytearray
def int2bytes(inp,size):
    out = []   # init
    for i in range(size):
        out.append(inp % 2)
        inp >>= 1
    out.reverse()	

    return np.array(out)

# bytearray to int    
def bytes2int(bytes):
    result = 0
    for b in bytes:
        result = result * 2 + int(b)
    return result

# bytearray to int    
def arr2int(arr):
    result = 0
    for i in range(len(arr)):
        if arr[len(arr) - i - 1] == 1:
            return (len(arr) - i)
    return result

#
def sumup(arr):
    count = 0
    for item in arr:
        if item == 1:
            count+=1
    return count

#
def ip_arr(arr,dim):
    shape =  arr.reshape((dim,int(len(arr)/dim)))	
    out = np.zeros((dim,),np.int)
    for i in range(len(out)):
        out[i] = arr2int(shape[i])
    
    return out

#
def ip_1darr(arr,dim):
    arr[arr==-1] = 0
    shape =  arr.reshape((dim,len(arr)/dim))	
    out = np.zeros((dim,),np.int)
    for i in range(len(out)):
        print(shape[i])
        out[i] = bytes2int(shape[i])
    
    return out

#
def ip_2darr(arr,dim):
    outs = np.zeros((arr.shape[0],dim),np.int)
    for i in range(arr.shape[0]):
        out = ip_1darr(arr[i],dim)
        outs[i,:] = out[:]
    return outs
#
def ip_2darr_(arr,dim):
    outs = np.zeros((arr.shape[0],dim),np.int)
    for i in range(arr.shape[0]):
        out = ip_arr(arr[i],dim)
        outs[i,:] = out[:]
    return outs
#facial landmarks to 
def fpoints2feature(points, filename = None):
    features = []
    #face
    face_w = distance(points[0],points[16])
    face_h = distance(points[27],points[8])
    #eyebrow
    leyebrow_w = distance(points[17],points[21])	
    reyebrow_w = distance(points[22],points[26])
    #eye
    leye_w = distance(points[36],points[40])
    leye_h1 = distance(points[37],points[41])
    leye_h2 = distance(points[38],points[40])
    reye_w = distance(points[42],points[45])
    reye_h1 = distance(points[43],points[47])
    reye_h2 = distance(points[44],points[46])
    #dis between eyebrow and eye
    leye_leyebrow_h1 = distance(points[17],points[36])
    leye_leyebrow_h2 = distance(points[18],points[37])
    leye_leyebrow_h3 = distance(points[19],points[38])
    leye_leyebrow_h4 = distance(points[20],points[39])
    reye_reyebrow_h1 = distance(points[22],points[42])
    reye_reyebrow_h2 = distance(points[23],points[43])
    reye_reyebrow_h3 = distance(points[24],points[44])
    reye_reyebrow_h4 = distance(points[25],points[45])				
    #mouse
    mouth_w1= distance(points[49],points[53])
    mouth_w2= distance(points[48],points[54])
    mouth_w3= distance(points[59],points[55])
    mouth_h1= distance(points[50],points[58])
    mouth_h2= distance(points[51],points[57])
    mouth_h3= distance(points[52],points[56])
    lmouth_h1=distance(points[49],points[8])
    lmouth_h2=distance(points[48],points[8])
    lmouth_h3=distance(points[59],points[8])
    rmouth_h1=distance(points[53],points[8])
    rmouth_h2=distance(points[54],points[8])
    rmouth_h3=distance(points[56],points[8])  
    #dis between upper lip and lower lip
    lips_h1= distance(points[61],points[67])
    lips_h2= distance(points[62],points[66])
    lips_h3= distance(points[63],points[65])
    #--->
    features.append(5 * (leyebrow_w + reyebrow_w))
    features.append(5 * (leyebrow_w + reyebrow_w))
    features.append(5 * (leye_w + reye_w))
    features.append(5 * (leye_w + reye_w))
    features.append(100 * (leye_h1 + leye_h2 + reye_h1 + reye_h2))
    features.append(100 * (leye_h1 + leye_h2 + reye_h1 + reye_h2))
    features.append(100 * (leye_h1 + leye_h2 + reye_h1 + reye_h2))
    features.append(100 * (leye_h1 + leye_h2 + reye_h1 + reye_h2))
    features.append(10 * (leye_leyebrow_h1 + leye_leyebrow_h2 + leye_leyebrow_h3 + leye_leyebrow_h4))	
    features.append(10 * (leye_leyebrow_h1 + leye_leyebrow_h2 + leye_leyebrow_h3 + leye_leyebrow_h4))
    features.append(10 * (leye_leyebrow_h1 + leye_leyebrow_h2 + leye_leyebrow_h3 + leye_leyebrow_h4))
    features.append(10 * (leye_leyebrow_h1 + leye_leyebrow_h2 + leye_leyebrow_h3 + leye_leyebrow_h4))
    features.append(10 * (reye_reyebrow_h1 + reye_reyebrow_h2 + reye_reyebrow_h3 + reye_reyebrow_h4))
    features.append(10 * (reye_reyebrow_h1 + reye_reyebrow_h2 + reye_reyebrow_h3 + reye_reyebrow_h4))
    features.append(10 * (reye_reyebrow_h1 + reye_reyebrow_h2 + reye_reyebrow_h3 + reye_reyebrow_h4))
    features.append(10 * (reye_reyebrow_h1 + reye_reyebrow_h2 + reye_reyebrow_h3 + reye_reyebrow_h4))
    features.append(4 * (mouth_w1 + mouth_w2 + mouth_w3))
    features.append(4 * (mouth_w1 + mouth_w2 + mouth_w3))
    features.append(4 * (mouth_w1 + mouth_w2 + mouth_w3))
    features.append(4 * (mouth_h1 + mouth_h2 + mouth_h3))
    features.append(4 * (mouth_h1 + mouth_h2 + mouth_h3))
    features.append(4 * (mouth_h1 + mouth_h2 + mouth_h3))
    features.append(4 * (lmouth_h1 + lmouth_h2 + lmouth_h3))
    features.append(4 * (lmouth_h1 + lmouth_h2 + lmouth_h3))
    features.append(4 * (lmouth_h1 + lmouth_h2 + lmouth_h3))
    features.append(4 * (rmouth_h1 + rmouth_h2 + rmouth_h3))
    features.append(4 * (rmouth_h1 + rmouth_h2 + rmouth_h3))
    features.append(4 * (rmouth_h1 + rmouth_h2 + rmouth_h3))
    features.append(4 * (lips_h1 + lips_h2 + lips_h3))
    features.append(4 * (lips_h1 + lips_h2 + lips_h3))
    features.append(4 * (lips_h1 + lips_h2 + lips_h3))
    features = np.array(features)
    #normalize
    arr = np.zeros((len(features),cfg.FEATURE_SIZE),np.int)
    arr[:] = -1
    for i in range(len(features)):
        index =	int((features[i] / float(face_w + face_h)) * cfg.FEATURE_SIZE)
        if index >= cfg.FEATURE_SIZE:
            index = (cfg.FEATURE_SIZE  - 1)
        arr[i][index] = 1

    if  filename is not None:
        fullname = filename + '.dat'
        arr.tofile(fullname)
    #arr2Img
    img =  arr2img(arr)		
    return img,arr

#make samples
def makeSample():
    rootdir = './train/'
    subdirs = ['Negative','Neutral','Positive']  
    sampledir = rootdir + 'T/'

    for subdir in subdirs:
        classdir = rootdir + subdir + '/'

        avg = np.zeros((cfg.INPUT_SIZE,),np.int)
        count = 0
        for filename in os.listdir(classdir):
            signal = np.fromfile(classdir + filename, dtype=np.int)
            avg = avg + signal
            count += 1
        threshold = -count / 2
        avg[avg > threshold] = 1
        avg[avg <= threshold] = -1
        avg.tofile(sampledir + subdir + '.dat')
    
#arr to points
def arr2pts(arr):
    xs = []
    ys = []
    count = 0
    if  len(arr.shape) == 1:
        dim = int(math.sqrt(len(arr)))
    arr = arr.reshape((dim,dim))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if  arr[i][j] == 1:
                xs.append(i)		
                ys.append(j)
                count += 1
    pts = np.zeros((count,2),np.int)
    for i in range(count):
        pts[i][0] = xs[i]
        pts[i][1] = ys[i]
    return pts	
	
#resize
def resizeDim(arr,newdim = 50,filename = None):
    pts = arr2pts(arr)
    return pts2arr(pts,newdim)
