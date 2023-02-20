import cv2
import streamlit as st
from deep_list import *
import torch
import os
import time  # to simulate a real time data, time loop
from app_us import OD
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import container_functions as container_class
import vechiles as vec

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    st.set_page_config(layout="wide")
    st.title("Egabi AI Team ")
    st.subheader('Detection and Tracking Vechiles')
  
    st.sidebar.image('logo.png',use_column_width=True)

    inference_msg = st.empty()
    st.sidebar.title("Configrations" ,)


    source = ("Shhipping Container", "Other Vechiles")
    source_index = st.sidebar.selectbox("Choose Model", range(
        len(source)), format_func=lambda x: source[x])
    
    speed = st.sidebar.text_input("Video Speed", value=1, max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, \
            args=None, placeholder="values from 1,2,3,4", disabled=False, label_visibility="visible")
    
    if source_index == 0:
        input_source = st.sidebar.radio(
            "Select input source",
        ('Local video' , 'Webcam' ,'RTSP'))
        if input_source == 'Local video':
            container_class.local_video(src=source_index , speed = int(speed))
        elif input_source == 'Webcam':    
            container_class.web_cam(src=source_index, speed = int(speed))
        else:
            container_class.rtsp(src=source_index, speed = int(speed))
          
    elif source_index == 1:
        input_source = st.sidebar.radio(
            "Select input source",
        ('Local video',))
        if input_source == 'Local video':
            vec.local_video(speed=int(speed))      

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

