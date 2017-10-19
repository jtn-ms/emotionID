import numpy as np
import os
import cv2
import cv2.cv as cv
from skimage import transform as tf
from PIL import Image, ImageDraw
import threading
from time import ctime,sleep
import time
import sklearn
import matplotlib.pyplot as plt
import skimage
import sklearn.metrics.pairwise as pw
import triplet._init_paths
import triplet.config as cfg
from triplet.sampledata import sampledata
from utils.timer import Timer
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import argparse
import glob
from sklearn.metrics import confusion_matrix
import pandas as pd

####
####Define LandMarker
####
global feature_size
feature_size=cfg.LANDMARK_SIZE

class DeepLandMark(caffe.Net):
    """
    Recognizer extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, mean_file=None,
		 image_dims=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
		 raw_scale=255,
                 channel_swap=(2,1,0),
  		 input_scale=None):
	#set GPU mode
	caffe.set_mode_gpu()
	#init net
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean_file is not None:
	    proto_data = open(mean_file, "rb").read()
	    mean_blob = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
	    mean = caffe.io.blobproto_to_array(mean_blob)[0]
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if cfg.CHANNEL_SIZE != 1 and channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims
	
	self.face_cascade = cv2.CascadeClassifier(cfg.FACE_CASCADE_XML)

    #show landmark
    def show_landmark(self,imgfile):
	 	

    def getFaceMat(self,img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)		
	for (x,y,w,h) in faces:
    	    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    return roi_gray
	    
    #process frame
    def getLandMarks(self,img):
        resized =skimage.transform.resize(img,(self.image_dims[0], self.image_dims[1]))*255
	X=np.empty((1,1,self.image_dims[0],self.image_dims[1]))
    	X[0,0,:,:]=resized[:,:]
	perpoint=2
        out = self.forward_all(data=X)
	#extract feature
	feature = np.float64(out['fc2'])
	feature=np.reshape(feature,(perpoint,feature_size))
        return feature

    #process an image file
    def getLandMarks(self,imgfile):
        img=skimage.io.imread(imgfile,as_grey=True)
        resized =skimage.transform.resize(img,(self.image_dims[0], self.image_dims[1]))*255
	X=np.empty((1,1,self.image_dims[0],self.image_dims[1]))
    	X[0,0,:,:]=resized[:,:]
	perpoint=2
        out = self.forward_all(data=X)
	#extract feature
	feature = np.float64(out['fc2'])
	feature=np.reshape(feature,(perpoint,feature_size))
        return feature

if __name__ == '__main__':
    modeldir = './landmark/model/'
    modelname = '1_F'
    landmark = DeepLandMark(modeldir + modelname + '.prototxt',
    	       	            modeldir + modelname + '.caffemodel')

    model_accuracy = landmark.test_alex()
																																															

