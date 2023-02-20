from textwrap import fill
import torch
import numpy as np
import cv2 
from time import time
from datetime import datetime
import math
from PIL import ImageFont, ImageDraw, Image
import streamlit as st
import gc
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use( 'tkagg')
#plt.style.use('seaborn-whitegrid')
class OD:
    def __init__(self, capture_index, model_name,stframe,source_model,speed):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
													
        self.stframe = stframe
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.frame_count = []
        self.count = 0
        self.is_container = False
        self.left_corner = []
        self.last_container = 0
        self.source_model = source_model
        self.speed = speed

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        #cv2.VideoCapture(self.capture_index , cv2.CAP_FFMPEG)
        """
        #rtsp = "rtsp://186.1.16.32:1161/02010533754159250101?DstCode=01&ServiceType=1&ClientType=1&StreamID=1&SrcTP=2&DstTP=2&SrcPP=1&DstPP=1&MediaTransMode=0&BroadcastType=0&SV=1&Token=pVt4XpuQUFvFTND0UIc6ZCCe4OF0Dj/zOz7G+RKAix8=&DomainCode=da30d0eb264e47968184273537e16acf&UserId=12"
        cap = cv2.VideoCapture(self.capture_index)
        #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        assert cap is not None
        return cap

    def load_model(self, model_name):
        model = torch.hub.load("yolov5",'custom' ,model_name, source='local')
        model.conf = 0.80  
        model.iou = 0.50 
        model.classes = 0
        return model
    
    def check_if_all_zero(self,arr):
        for elem in arr:
            if elem != 0:
                return False
        return True 


    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        
        labels, cord ,confidence = results.xyxyn[0][:, -1], results.xyxy[0][:,:-1],results.xyxyn[0][:, -2]
        
        try:
            if labels.size()[0] == 0:
                self.frame_count.append(0)
                
            if labels.size()[0] == 1:
                if self.count == 0 or (self.left_corner[-2] - self.left_corner[-1] >= 150): 
                    self.is_container = True
                self.frame_count.append(1)
                self.left_corner.append(cord[0][0].item())
                #print(self.left_corner[-2] , self.left_corner[-1])
            """if self.left_corner[-1] < self.left_corner[-2]:
                print(self.left_corner)"""    
            if self.check_if_all_zero(self.frame_count[-10:]):
                if self.is_container == True:
                    self.count +=1
                    #self.left_corner = []
                    self.is_container = False
                             
        except :
            pass
        
        return labels, cord ,confidence

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord, confidence = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]) + " " +str([round(x.item(),2) for x in confidence][i]),\
                            (x1, y1), cv2.FONT_HERSHEY_SIMPLEX , 0.7 , (0, 234, 255), 2 ,cv2.LINE_AA)  
                
        return frame

   
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
        video_fps = cap.get(cv2.CAP_PROP_FPS),
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #print('w' + str(width) + "h" + str(height))
        # we are using x264 codec for mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('out2.mp4', apiPreference=0, fourcc=fourcc,
                                fps=video_fps[0], frameSize=(int(width), int(height)))
        pts_n = 100
        x = [0]
        y = [0]     
        #my_process = psutil.Process(os.getpid())
        #t_start = time.time()
        # ADD Egabi
        i = 0
        fig, ax = plt.subplots()
        (ops, ) = ax.plot(x, y, linestyle="--")
        ax.set_ylim(bottom=0)
        old = 0
        new = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if(frame is None):
                continue 
            i+=1
            if i % self.speed != 0:
                continue 
            results = self.score_frame(frame)
            # video Cropping
            frame = frame[300:,::]
            frame = self.plot_boxes(results, frame)
            
            if self.source_model == 0:
                with self.stframe.container():
                    kpi1,kpi2 = st.columns([20,8])
                    
                    with kpi1:
                        st.image(frame, channels="BGR", width=1000,use_column_width=False)

                    with kpi2:
                        new_title = '<p style="background-color: rgba(28, 131, 225, 0.1); \
                                        border: 1px solid rgba(28, 131, 225, 0.1); \
                                        padding: 5% 5% 5% 10%; \
                                        border-radius: 5px; \
                                        color: red; \
                                        font-size:40px;  \
                                        text-align: center;    \
                                        overflow-wrap: break-word;"> حاويات الشحن <br>{} </p> '.format(self.count)
                                        
                        st.markdown(new_title, unsafe_allow_html=True)

                
