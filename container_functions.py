import streamlit as st
from app_us import OD
from deep_list import *
import gc
import torch

# ------------------------- LOCAL VIDEO ------------------------------
def local_video(src , speed):
    video = st.sidebar.file_uploader("Select input video", type=["mp4", "avi"], accept_multiple_files=False)
    if video is not None:
            is_valid = True
            with st.spinner(text='Video...'):
                st.sidebar.video(video)
                with open(os.path.join("data", "videos", video.name), "wb") as f:
                    f.write(video.getbuffer())
                source = f'data/videos/{video.name}'
    else:
        is_valid = False
    if is_valid :    
        torch.cuda.empty_cache()
        gc.collect()
        if st.sidebar.button("Start tracking"):
            stframe = st.empty()
            detector = OD(capture_index=source, model_name="./two-class-model-huawei.pt",stframe=stframe , source_model = src , speed=speed)
            detector()


def web_cam(src,speed):
    if st.sidebar.button("Start tracking"):
        torch.cuda.empty_cache()
        gc.collect()
        stframe = st.empty()
        detector = OD(capture_index=0 , model_name="./two-class-model-huawei.pt",stframe=stframe, source_model = src, speed=speed)
        detector()


def rtsp(src,speed):
    rtsp_input = st.sidebar.text_input("IP Address", "rtsp://192.168.0.1")
    if st.sidebar.button("Start tracking"):
        torch.cuda.empty_cache()
        gc.collect()
        stframe = st.empty()
        detector = OD(capture_index = rtsp_input , model_name="./two-class-model-huawei.pt",stframe=stframe, source_model = src, speed=speed)
        detector()            

       


    