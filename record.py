#import Tkinter as tk
import os,sys
   
try:#if python2
    from Tkinter import *
    #import Tkinter	
    import tkMessageBox
    print('current python version is 2')
    py_ver = 2
except ImportError:#else python3
    from tkinter import *
    #import tkinter
    from tkinter import messagebox
    print('current python version is 3')
    py_ver = 3

import numpy as np
import cv2
import config as cfg

from PIL import Image, ImageTk

#import caffe
#from caffe.proto import caffe_pb2

#from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib

from random import randint

import util as utl

(cv_major_ver, cv_minor_ver, cv_subminor_ver) = (cv2.__version__).split('.')

#webcam + save video parameter
global cap
global width,height
global face_cascade
global detector
global predictor

#
cap = cv2.VideoCapture(0)
#
width, height = cfg.WIDTH, cfg.HEIGHT
fps=20
#cascade
face_cascade = cv2.CascadeClassifier(cfg.FACE_CASCADE_XML)
#dlib landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cfg.LANDMARK_PREDICTOR)

'''
global fps
global fourcc
global writer

if int(major_ver) < 3 :
    fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
    writer = cv2.VideoWriter('record.AVI',	fourcc, fps, (width,height), 1)
else:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('record.AVI',	fourcc, fps, (width,height), True)
'''


class StopWatch(object):
    def __init__(self):	
        self.start = time.time()
        self.stop = time.time()
        self.isRUNNING = False
    def doStart(self):
        self.start = time.time()
        self.isRUNNING = True
    def doStop(self):
        if self.isRUNNING == False:
            return	
        self.stop = time.time()
        self.isRUNNING = False
    def doReport(self):
        if self.isRUNNING:
            return '%.2f' % (time.time() - self.start)
        else:
            return '%.2f' % (self.stop - self.start)
    def checkRunning(self):
        return self.isRUNNING

#Maximize the Screen
class FullScreenApp(object):
    def __init__(self, master, **kwargs):
     self.master=master
     pad=3
     master.configure(background='black')
     self._geom='200x200+0+0'
     master.geometry("{0}x{1}+0+0".format(master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
     master.bind('<Escape>',self.toggle_geom)
     master.wm_protocol("WM_DELETE_WINDOW", self.on_closing)
     #master.bind('<Escape>', lambda e: master.quit())
     master.bind('<Escape>', self.on_close)
     master.bind('t',self.T_Pressed)
     master.bind('f',self.F_Pressed)
     master.bind("1",self.One_Pressed)#<BUTTON-1>-MOUSE LEFT BUTTON
     master.bind("2",self.Two_Pressed)
     master.bind("7",self.show_lview)
     master.bind("8",self.hide_lview)
     master.bind("9",self.show_rview)
     master.bind("0",self.hide_rview)
     master.bind("<space>", self.Space_Pressed)
     master.bind("<Return>",self.Enter_Pressed)
	#space between button and picture
     self.margin = Frame(self.master,width=100,height=20)
     self.margin.configure(background='black')
     self.margin.pack()
     #text label
     self.question = Label(self.master, text = '', bg = 'black', fg = 'white')
     self.question.config(font=("Courier", 18))
     self.question.config(width=200)
     self.question.pack()
     self.instruction = Label(self.master, text = cfg.PREPARATION, bg = 'black', fg = 'white')
     self.instruction.config(font=("Courier", 12))
     self.instruction.config(width=200)
     self.instruction.pack()
     #space between label and picture
     self.frame = Frame(self.master,width=100,height=50)
     self.frame.configure(background='black')
     #self.frame.pack()

     #state info or temp
     self.Img = np.zeros((height,width,3), np.uint8)
     self.cameraoffline = True
     self.featurepoints = np.zeros((cfg.LANDMARK_SIZE,2),np.int)
     self.Stages = ['Preparing','Training','Testing']
     self.Sessions = ['Neutral','Positive','Negative']
     self.SessionStarted = False
     self.CountingStarted = False
     self.CorrectPosed = False
     self.currentSessionID = 0
     self.currentStageID =  0
     self.currentQuestionID = 0
     self.snapshotID = 0
     self.stopWatch = StopWatch()
     #make dirs
     self.createOutDirs()
     
     #
     self.closing = False
     
     #show input frame
     self.lview = Label(self.master)
     self.lview.pack(side=LEFT)
     self.show_inputframe()
     #show output frame
     self.rview = Label(self.master)
     self.rview.pack(side=RIGHT)
     self.show_outputframe()
     
     #######################################################################################
    #########################################################################################
     #######################################################################################
    def makeDir(self,dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
	
    def createOutDirs(self):
        self.makeDir('./train')
        for session in self.Sessions:	
            self.makeDir('./train/' + session)

    def changeQuestion(self,question=''):
        if question != '':
            question = cfg.PREFIX_Q + question
        self.question['text'] = question
        print(question)
	
    def changeInstruction(self,instruction):
        self.instruction.config(text=instruction)
    #############################################
    def hide_lview(self,event = None):
        self.lview.pack_forget()

    def show_lview(self,event = None):
        self.lview.pack(side=LEFT)

    def hide_rview(self,event = None):
        self.rview.pack_forget()

    def show_rview(self,event = None):
        self.rview.pack(side=RIGHT)
     #######################################################################################
    ###########################KEY EVENT HANDLE##############################################
     ###########################################
    def Space_Pressed(self,event):
        if  self.Stages[self.currentStageID] == 'Preparing' and self.CorrectPosed:
            self.currentStageID = 1
            self.hide_lview()
            self.changeInstruction(cfg.INSTRUCTION1)   
            return
        if  self.currentQuestionID != 0 or\
            self.Stages[self.currentStageID] != 'Training' or \
	        self.SessionStarted:
            return
        self.SessionStarted = True	
        #print cfg.PREFIX_I,self.Sessions[self.currentSessionID],'State',self.Stages[self.currentStageID],' Session Started'
        self.changeInstruction(cfg.INSTRUCTION2)
        self.changeQuestion(cfg.QUESTIONS[self.currentSessionID][self.currentQuestionID])
 	
    def Enter_Pressed(self,event):
        if  self.SessionStarted == False or\
        self.Stages[self.currentStageID] != 'Training':
            return
        self.stopWatch.doStart()
        self.changeInstruction(cfg.INSTRUCTION3s[self.currentSessionID])

    def T_Pressed(self,event):
        # pass only when neutral state is being trained and training session already started
        if self.SessionStarted == False or \
	       self.stopWatch.checkRunning() == False or \
	       self.Stages[self.currentStageID] != 'Training' or \
	       self.currentSessionID != 0:
           return
        # record time, save Snapshot,print log
        self.stopWatch.doStop()
        #cv2.imwrite('./train/' + self.Sessions[self.currentSessionID] + '/' + \
        #			str(self.currentQuestionID) + '_' + self.stopWatch.doReport() +'.png',self.Img)
        utl.fpoints2feature(self.featurepoints,'./train/' + self.Sessions[self.currentSessionID] + '/' +str(self.currentQuestionID))
        self.snapshotID += 1
        print('True')
        # change question,instruction
        self.currentQuestionID += 1
        #whether the current session is over or not
        if  self.currentQuestionID == cfg.QUESTION_COUNT:   
            self.SessionStarted = False
            self.currentQuestionID = 0
            self.currentSessionID += 1
            self.changeQuestion()
            #if training is over
            if  self.currentSessionID == 3:
                self.currentSessionID = 0
                self.currentStageID += 1
                self.changeInstruction(cfg.INSTRUCTION4)
            else:	
                self.changeInstruction(cfg.INSTRUCTION1)
        else:
            self.changeQuestion(cfg.QUESTIONS[self.currentSessionID][self.currentQuestionID])
            self.changeInstruction(cfg.INSTRUCTION2)

    def F_Pressed(self,event):
        self.stopWatch.doStop()
        self.changeQuestion(cfg.QUESTIONS[self.currentSessionID][randint(0,cfg.QUESTION_COUNT)])
        self.changeInstruction(cfg.INSTRUCTION2)
        print('False')

    def  One_Pressed(self,event):
        # pass only when postive/negative state is being trained and training session already started
        if self.SessionStarted == False or \
	       self.stopWatch.checkRunning() == False or \
	       self.Stages[self.currentStageID] != 'Training' or \
	       self.currentSessionID == 0:
               return
        # record time, save Snapshotl, print log, turn off timing
        self.stopWatch.doStop()
        #cv2.imwrite('./train/' + self.Sessions[self.currentSessionID] + '/' + \
        #			str(self.currentQuestionID) + '_' + self.stopWatch.doReport() +'.png',self.Img)
        utl.fpoints2feature(self.featurepoints,'./train/' + self.Sessions[self.currentSessionID] + '/' +str(self.currentQuestionID))	
        self.snapshotID += 1
        print('I REMEMBER')
        # change question,instruction
        self.currentQuestionID += 1 
        #whether the current session is over or not
        if  self.currentQuestionID == cfg.QUESTION_COUNT:   
            self.currentQuestionID = 0
            self.currentSessionID += 1
            self.SessionStarted = False
            self.changeQuestion()
            #if training is over
            if self.currentSessionID == 3:
                self.currentSessionID = 0
                self.currentStageID += 1
                self.changeInstruction(cfg.INSTRUCTION4)
            else:
                self.changeInstruction(cfg.INSTRUCTION1)
        else:
            self.changeQuestion(cfg.QUESTIONS[self.currentSessionID][self.currentQuestionID])
            self.changeInstruction(cfg.INSTRUCTION2)
	 	 
    def Two_Pressed(self,event):
        self.stopWatch.doStop()
        self.changeQuestion(cfg.QUESTIONS[self.currentSessionID][randint(0,cfg.QUESTION_COUNT)])
        self.changeInstruction(cfg.INSTRUCTION2)
        print('I DONT REMEMBER')
     ###################################################################################
    ###########################FRAME HANDLE##############################################
     ###################################################################################
    def show_outputframe(self):
        if self.cameraoffline or np.all(self.featurepoints == 0):
            return
        gray,arr = utl.fpoints2feature(self.featurepoints)
        tmp = cv2.resize(gray, (width,height))
        cv2image = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.rview.imgtk = imgtk
        self.rview.configure(image=imgtk)
        self.rview.after(10, self.show_outputframe)
	        
    def show_inputframe(self):
        if self.closing:
            self.cameraoffline = True
            return
        online, frame = cap.read()
        if online == False:
            self.cameraoffline = True
            return
        self.cameraoffline = False
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (width,height))
        ##save video
        #writer.write(frame)
        ##save image
        self.Img = frame.copy()
        #cv2.imshow('in',frame)
        #cv2.waitKey(1)
        #LandMark
        self.show_landmark(frame)
        #Red mark
        cv2.circle(frame, (cfg.FOCAL_POINT[0],cfg.FOCAL_POINT[1]), 50, (0,0,255),3)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lview.imgtk = imgtk
        self.lview.configure(image=imgtk)
        self.lview.after(10, self.show_inputframe)

    def ready4Training(self,img,shape,faceindex,x,y):
        allowed = 12	
        ##
        move_right = cfg.FOCAL_POINT[0] - shape.item((27,0))  
        move_down = cfg.FOCAL_POINT[1] - shape.item((27,1))
        rotated,tilted = 0,0
        for i in range(6):
            rotated += (shape[i + 36][1] - shape[i + 42][1]) 
        rotated /= 6
        tilted = (shape[27][0] - shape[0][0]) - (shape[16][0] - shape[27][0])
        
        msg = ''
        if move_right > allowed:
            msg += 'move right, '
        if (move_right + allowed) < 0:
            msg += 'move left, '
        if move_down > allowed:
            msg += 'move down, '
        if (move_down + allowed) < 0:
            msg += 'move up, '
        if rotated > allowed / 4:
            msg +=  'move clockwise, '
        if (rotated + allowed / 4) < 0:
            msg += 'move counterclockwise, '
        if tilted > allowed:
            msg +=  'rotate to the left'
        if (tilted + allowed) < 0:
            msg += 'rotate to the right'
            
        if msg == '':  
            cv2.putText(img, "Face #{} OK! Hold it!".format(faceindex + 1), (20, 50),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.CorrectPosed = True
        else:
            cv2.putText(img, msg, (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)		
            self.CorrectPosed = False

    def drawLine(self,img,shape,here,there,isClosed=False):
        for i in range(here,there):
            cv2.line(img,(shape[i][0],shape[i][1]),(shape[i + 1][0],shape[i + 1][1]),(0, 255, 0),1)		
        if isClosed:
            cv2.line(img,(shape[here][0],shape[here][1]),(shape[there][0],shape[there][1]),(0, 255, 0),1)

    def drawContours(self,img,coords):
        self.drawLine(img,coords,0,16)
        self.drawLine(img,coords,17,21)
        self.drawLine(img,coords,22,26)
        self.drawLine(img,coords,27,30)
        self.drawLine(img,coords,30,35,True)
        self.drawLine(img,coords,36,41,True)
        self.drawLine(img,coords,42,47,True)
        self.drawLine(img,coords,48,59,True)
        self.drawLine(img,coords,60,67,True)
        cv2.line(img,(coords[48][0],coords[48][1]),(coords[60][0],coords[60][1]),(0, 255, 0),1)
        cv2.line(img,(coords[54][0],coords[54][1]),(coords[64][0],coords[64][1]),(0, 255, 0),1)

    def show_landmark(self,img):
        frame = imutils.resize(img, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # calculate ratio
        x_ratio = float(width) / np.shape(gray)[1]
        y_ratio = float(height) / np.shape(gray)[0]
        # map
        coords = np.zeros((cfg.LANDMARK_SIZE, 2), dtype='int')
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            self.featurepoints = shape
            (x_, y_, w_, h_) = face_utils.rect_to_bb(rect)
            #x, y, w, h = int(x_ * x_ratio), int(y_ * y_ratio), int(w_ * x_ratio), int(h_ * y_ratio)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image   
            for j in range(0, cfg.LANDMARK_SIZE):
                coords[j] = (int(x_ratio * shape[j][0]), int(y_ratio * shape[j][1]))	        
                cv2.circle(img, (coords[j][0], coords[j][1]), 1, (0, 0, 255), -1)
            self.drawContours(img,coords)
            self.ready4Training(img,coords,i,int(x_ * x_ratio),int(y_ * y_ratio))		    
        return coords

     #####################################################################################
    ###########################WINDOWS HANDLE##############################################
     #####################################################################################    
    def toggle_geom(self,event):
        print('toggle clicked')
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom


    def on_closing(self):
        self.closing = True
        print('close clicked')
        if py_ver == 3:
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                self.master.destroy()
        else:
            if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
                self.master.destroy()
    
    def on_close(self,event):
        self.closing = True
        print('escape clicked')
        self.master.withdraw() # if you want to bring it back
        sys.exit() # if you want to exit the entire thing        

if __name__ == '__main__':
    root = Tk()#Toplevel()
    app = FullScreenApp(root)
    root.mainloop()
