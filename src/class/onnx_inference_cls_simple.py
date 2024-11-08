import onnxruntime as rt
import cv2
import math
import numpy
from mmdet.core.post_processing import multiclass_nms
import os

import argparse
import os.path as osp
import warnings

import numpy as np
import onnx
import torch
from mmcv import Config, DictAction

from mmdet.core.export import build_model_from_cfg, preprocess_example_input
from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
from pathlib import Path
import copy


def predclas(sess, img):
    input_name = sess.get_inputs()[0].name

    img_roi = copy.deepcopy(img)
    longer_side = max(img_roi.shape[0], img_roi.shape[1])
    resize_factor = 128.0 / longer_side
    resize_height = int(np.round(img_roi.shape[0] * resize_factor))
    resize_width = int(np.round(img_roi.shape[1] * resize_factor))
    print("resize_height: ", resize_height, "resize_width: ", resize_width)
    img_roi = cv2.resize(img_roi, (resize_width, resize_height), dst=None, interpolation=cv2.INTER_LINEAR)
    total_pad = 128 - min(img_roi.shape[1], img_roi.shape[0])
    if img_roi.shape[1] < img_roi.shape[0]:
        img_roi = cv2.copyMakeBorder(img_roi, 0, 128 - img_roi.shape[0], int(total_pad / 2),
                                     total_pad - int(total_pad / 2), cv2.BORDER_CONSTANT, 0).astype(np.float32)[
            None, ...]
    else:
        img_roi = cv2.copyMakeBorder(img_roi, int(total_pad / 2),
                                     total_pad - int(total_pad / 2), 0, 128 - img_roi.shape[1], cv2.BORDER_CONSTANT,
                                     0).astype(np.float32)[None, ...]
    img_roi = img_roi[:, :, :, [2, 1, 0]]
    img_roi = img_roi.transpose(0, 3, 1, 2)
    img_roi = (img_roi - np.array([0, 0, 0], dtype=float)[None, :, None, None]) / np.array(
        [255., 255., 255.], dtype=float)[None, :, None, None]

    pred_color, pred_shape, pred_toward = sess.run(
        None, {input_name: img_roi.astype(np.float32)})
    result = {}

    result['color_score'] = pred_color[0]
    result['shape_score'] = pred_shape[0]
    result['toward_score'] = pred_toward[0]
    result['color_label'] = np.argmax(pred_color[0])
    result['shape_label'] = np.argmax(pred_shape[0])
    result['toward_label'] = np.argmax(pred_toward[0])

    for key in result.keys():
        if type(result[key]) == numpy.int64:
            result[key] = int(result[key])
        else:
            result[key] = list(result[key])
            result[key] = list(map(str, result[key]))
        # print(key," is ",result[key])
    return result


def plotimg(img, results):
    SHAPE_CLASSES = [
        'circle',
        'uparrow',
        'downarrow',
        'leftarrow',
        'rightarrow',
        'returnarrow',
        'bicycle',
        'others',
    ]

    COLOR_CLASSES = [
        'red',
        'yellow',
        'green',
        'black',
        'others',
        'unknow'

    ]

    TOWARD_CLASSES = [
        'front',
        'side',
        'backside',
        'unknow'
    ]
    SAMPLE_CLASSES = ['simple', 'complex']
    for result in results:
        # bbox = list(map(int, result['bbox']))
        # bbox = list(result['bbox'])
        bbox = result['bbox']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (108, 76, 255), 2)
        # i=1
        color = result["boxcolor"]
        shape = result["boxshape"]
        toward = result["toward_orientation"]
        # simple = result["simplelight"]
        # numsubcolor=result['subcolor_label']
        # for txt in result.keys():
        # print(txt)
        # print(type(txt))
        # print(txt)
        # if txt!="bbox" and txt.split("_")[-1]!="score":
        cv2.putText(img, COLOR_CLASSES[color], (bbox[0], bbox[1] + bbox[3] + 10), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (108, 76, 255))
        cv2.putText(img, SHAPE_CLASSES[shape], (bbox[0], bbox[1] + bbox[3] + 30), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (108, 76, 255))
        cv2.putText(img,TOWARD_CLASSES[toward],(bbox[0],bbox[1]+bbox[3]+50),cv2.FONT_HERSHEY_DUPLEX,0.8,(108,76,255))
        # cv2.putText(img, SAMPLE_CLASSES[simple], (bbox[0], bbox[1] + bbox[3] + 70), cv2.FONT_HERSHEY_DUPLEX, 0.8,
        #             (108, 76, 255))

    return img


#cls_onnx_file = "/mnt/ve_share/lijixiang/HE Zijian/traffic/work-dir/trafficlight/class_es11_daytime/cla_xmt_2_20230912.onnx"
cls_onnx_file = "/mnt/ve_share/lijixiang/HE Zijian/traffic/stage2/saves/stage2_1024.onnx"
data_path = '/mnt/ve_share/lijixiang/HE Zijian/traffic/eval/saves/infer_crop'
save_path = '/mnt/ve_share/lijixiang/HE Zijian/traffic/eval/saves'



if __name__ == "__main__":
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sessclas = rt.InferenceSession(cls_onnx_file, providers=EP_list)

    image_list = os.listdir(os.path.join(data_path,''))
    image_list = [i for i in image_list if i.endswith('.jpg')]
    image_list.sort()
    save = True

    color_num = 0
    red_num = 0
    yellow_num = 0
    green_num = 0
    black_num = 0
    color_tp_num = 0
    red_tp_num = 0
    yellow_tp_num = 0
    green_tp_num = 0
    black_tp_num = 0
    for index1 in range(len(image_list)):
        input_path = os.path.join(data_path,image_list[index1])
        im = cv2.imread(input_path)
        total_color = 5
        total_shape = 7
        if 1:
            cropimg = im
            result = predclas(sessclas, cropimg)
            print(image_list[index1],result['color_score'])
            color_num += 1
            if total_color == 0:
                red_num += 1
            if total_color == 1:
                yellow_num += 1
            if total_color == 2:
                green_num += 1
            if total_color == 3:
                black_num += 1
            if total_color == result["color_label"]:
                color_tp_num += 1
                if total_color == 0:
                    red_tp_num += 1
                if total_color == 1:
                    yellow_tp_num += 1
                if total_color == 2:
                    green_tp_num += 1
                if total_color == 3:
                    black_tp_num += 1

                # exit()
    print("color_tp_num ", color_tp_num)
    print("color_num ", color_num)
    print("color right rate ", color_tp_num/color_num)

    print("red_tp_num ", red_tp_num)
    print("red_num ", red_num)
    print("red right rate ", red_tp_num/red_num)

    print("yellow_tp_num ", yellow_tp_num)
    print("yellow_num ", yellow_num)
    print("yellow right rate ", yellow_tp_num/yellow_num)

    print("green_tp_num ", green_tp_num)
    print("green_num ", green_num)
    print("green right rate ", green_tp_num/green_num)
