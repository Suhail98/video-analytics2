import cv2
import streamlit as st
from deep_list import *
import torch
import gc


def local_video(speed):
    uploaded_file = st.sidebar.file_uploader("Select input video", type=["mp4", "avi"], accept_multiple_files=False)
    if uploaded_file is not None:
            torch.cuda.empty_cache()
            gc.collect()
            is_valid = True
            with st.spinner(text='Video...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                source = f'data/videos/{uploaded_file.name}'
    else:
        is_valid = False
    if is_valid :    
        torch.cuda.empty_cache()
        gc.collect()
        if st.sidebar.button("Start tracking"):
            stframe = st.empty()
            #classes=[2, 3, 5, 7] ,
            detect(source=source, stframe=stframe , classes=[2, 3, 5, 7] , spped=speed)


# def web_cam():
#     if st.sidebar.button("Start tracking"):
#         torch.cuda.empty_cache()
#         gc.collect()
#         stframe = st.empty()
#         detect(source="0", stframe=stframe , classes=[2, 3, 5, 7] , spped=2)


# def rtsp():
#     rtsp_input = st.sidebar.text_input("IP Address", "rtsp://192.168.0.1")
#     if st.sidebar.button("Start tracking"):
#         torch.cuda.empty_cache()
#         gc.collect()
#         stframe = st.empty()
#         detect(source=rtsp_input, stframe=stframe , classes=[2, 3, 5, 7] , spped=2)            
