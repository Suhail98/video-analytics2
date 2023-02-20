import argparse
import time
from pathlib import Path
import streamlit as st
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, './yolov5') # Path for internal module without changing base
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.general import set_logging
from yolov5.utils.plots import Annotator, colors, save_one_box, plot_one_box
from yolov5.utils.torch_utils import select_device, time_sync
import json

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from graphs import bbox_rel,draw_boxes
from collections import Counter

import psutil
import subprocess

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)
    return dict

@torch.no_grad()
def detect(weights=ROOT / 'yolov5/yolov5n.pt',  # model.pt path(s)
        source=ROOT / 'yolov5/data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'yolov5/data/coco128.yaml',  # dataset.yaml path
        stframe=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.70,  # confidence threshold
        iou_thres=0.50,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,
        display_labels=False,
        config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml", #Deep Sort configuration
        spped = 1
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    ## initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    

    
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        #view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    t0 = time.time()
    
    dt, seen = [0.0, 0.0, 0.0], 0
    prev_time = time.time()
    selected_names = names.copy()
    global_graph_dict = dict()
    global_drift_dict = dict()
    test_drift = []
    frame_num = -1
    poor_perf_frame_counter=0
    mapped_ = dict()
    min_FPS = 10000
    max_FPS = -1
    und_dic = dict()
    final_frames_counting = []
    check_id = []
    count_check = 0
    count_list_appear_once = []
    final_final = []

    car_count = 0
    truck_count = 0
    motorcycle_count = 0
    bus_count = 0
    dic = {
        2 : 'car',
        3 : 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    for path, im, im0s, vid_cap, s in dataset:
        frame_num = frame_num+1
        if frame_num % spped == 0:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # detectin only
            dt[2] += time_sync() - t3

            # Process predictions
            class_count = 0
            
            drift_dict = dict()
            
            for i, det in enumerate(pred):  # per image
                
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    names_ = []
                    cnt = []

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        names_.append(names[int(c)])
                        cnt.append(int(n.detach().cpu().numpy()))
                    mapped_.update(dict(zip(names_, cnt)))
                    
                    global_graph_dict = Counter(global_graph_dict) + Counter(mapped_)

                    bbox_xywh = []
                    confs = []
                    cls_lst = []
                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        # print("class : " , cls)
                        # print("frame : " , frame_num)
                        cls_lst.append(cls.item())
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])
                        # print("conf : {0}, conf_t : {1}".format(conf, conf_thres))
                        
                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)
                    #print("fss" , confss)
                    # Pass detections to deepsort
                    ########################################################
                    outputs = deepsort.update(xywhs, confss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        # print("Outputs :", outputs)
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        #print("ID : " , identities)  # Id of objects
                        draw_boxes(im0, bbox_xyxy)

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]  # need id yasta
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_num, identity, bbox_left,
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                    # Write results Label
                    for *xyxy, conf, cls in reversed(det):
                        #print("class : " , cls)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img or display_labels:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    ## Our code working Sebaie & Mohsen
                    if len(outputs) > 0:
                        detc_list = det[:, -1].tolist()
                        id_list = outputs[:, -1].tolist()
                        if len(detc_list) == len(id_list):
                            und_dic[frame_num] = list(zip(detc_list , id_list))
                            #print(list(zip(detc_list , id_list)))
                            for n in list(zip(detc_list , id_list)):
                                if n in final_frames_counting or n[1] in check_id:
                                    count_list_appear_once.append(n[1])
                                    continue
                                else:
                                    #print('from else')
                                    check_id.append(n[1])
                                    final_frames_counting.append(n)
                                    count_check +=1       
                        else:
                            continue
                     
                else:
                    deepsort.increment_ages()
                #################################################################    
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
                
                curr_time = time.time()
                fps_ = curr_time - prev_time
                fps_ = round(1/round(fps_, 3),1)
                prev_time = curr_time
                
                #stframe.image(im0, channels="BGR", use_column_width=True)

                if count_check > 0:
                    if dic[final_frames_counting[-1][0]] == 'car':
                        car_count +=1
                    elif dic[final_frames_counting[-1][0]] == 'motorcycle':
                        motorcycle_count +=1    
                    elif dic[final_frames_counting[-1][0]] == 'bus':
                        bus_count +=1 
                    else:
                        truck_count +=1 
                count_check = 0

                ####### streamlit Code
                with stframe.container():
                    kpi1, kpi2, kpi3 , kpi4 = st.columns(4)

                    with kpi1:
                            new_title = '<p style="background-color: rgba(28, 131, 225, 0.1); \
                                            border: 1px solid rgba(28, 131, 225, 0.1); \
                                            padding: 5% 5% 5% 10%; \
                                            border-radius: 5px; \
                                            color: red; \
                                            font-size:40px;  \
                                            text-align: center;    \
                                            overflow-wrap: break-word;"> شاحنة <br>{} </p> '.format(truck_count)
                                            
                            st.markdown(new_title, unsafe_allow_html=True)


                    with kpi2:
                            new_title = '<p style="background-color: rgba(28, 131, 225, 0.1); \
                                        border: 1px solid rgba(28, 131, 225, 0.1); \
                                        padding: 5% 5% 5% 10%; \
                                        border-radius: 5px; \
                                        color: red; \
                                        font-size:40px;  \
                                        text-align: center;    \
                                        overflow-wrap: break-word;"> دراجة بخارية <br>{} </p> '.format(motorcycle_count)
                                        
                            st.markdown(new_title, unsafe_allow_html=True)
                                

                    with kpi3:
                            new_title = '<p style="background-color: rgba(28, 131, 225, 0.1); \
                                        border: 1px solid rgba(28, 131, 225, 0.1); \
                                        padding: 5% 5% 5% 10%; \
                                        border-radius: 5px; \
                                        color: red; \
                                        font-size:40px;  \
                                        text-align: center;    \
                                        overflow-wrap: break-word;"> حافلة <br>{} </p> '.format(bus_count)
                                        
                            st.markdown(new_title, unsafe_allow_html=True)

                    with kpi4:
                            new_title = '<p style="background-color: rgba(28, 131, 225, 0.1); \
                                        border: 1px solid rgba(28, 131, 225, 0.1); \
                                        padding: 5% 5% 5% 10%; \
                                        border-radius: 5px; \
                                        color: red; \
                                        font-size:40px;  \
                                        text-align: center;    \
                                        overflow-wrap: break-word;"> ملاكى <br>{} </p> '.format(car_count)
                                        
                            st.markdown(new_title, unsafe_allow_html=True)
                    v1, v2, v3  = st.columns([2,8,2])
                    with v2 :
                        st.image(im0, channels="BGR", width=1000,use_column_width=False)
        else:
            continue

    
    for i in final_frames_counting:
        if i[1] in count_list_appear_once:
            final_final.append(i)
        else:continue    
                
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
    if vid_cap:
        vid_cap.release()

    return final_final