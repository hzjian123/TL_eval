from pathlib import Path
import os
import sys
import glob
from cv2 import copyTo
import numpy as np
import argparse
import json
import math

from torch import float32

debug_os = "qnx"
# debug_os = "x86"

# 使用iou方式该变量起作用
iou_debug_thres = 0.5

# 过滤条件-bbox像素小于该值过滤
traffic_light_detect_filter_bbox = 4

# 是否使用自定义iou对比
# calculate_method = ""
calculate_method = "iou"

# 当标注文件被完全过滤掉是否也过滤掉检测结果对应的文件内容
# False=不过滤过检内容，True=过滤过检内容
filter_infer_txt = True

# 是否根据过滤条件过滤检测结果
# False=不过滤检测结果，True=过滤检测结果
filter_infer_bboxs = True

class BBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def iou_calculate(a, b):

    assert isinstance(a, BBox)
    assert isinstance(b, BBox)

    area_a = a.w * a.h
    area_b = b.w * b.h

    w = min(b.x+b.w, a.x+a.w) - max(a.x, b.x)
    h = min(b.y+b.h, a.y+a.h) - max(a.y, b.y)

    if w <= 0 or h <= 0:
        return 0

    area_c = w * h

    return area_c / (area_a + area_b - area_c)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def get_files(path, suffix='*'):
    path_ab = str(Path(path).resolve())
    if '*' in path_ab:
        files = sorted(glob.glob(path_ab, recursive=True))
    elif os.path.isdir(path_ab):
        files = sorted(glob.glob(os.path.join(path_ab, "*." + suffix)))
    else:
        raise Exception(f'Error: {path_ab} does not exist')
    return files

def iou_bbox(label_bbox, infer_bbox, thres=0.5):
    label_bbox_tmp = BBox(label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3])
    infer_bbox_tmp = BBox(infer_bbox[0], infer_bbox[1], infer_bbox[2], infer_bbox[3])
    if iou_calculate(label_bbox_tmp, infer_bbox_tmp) >= thres:
        return True
    return False


def iou_bboxs(label_bboxs, infer_bboxs, thres=0.5):
    if len(label_bboxs) != 0:
        label_bboxs = label_bboxs[0]
    if len(infer_bboxs) != 0:
        infer_bboxs = infer_bboxs[0]
    count = 0
    for bbox in infer_bboxs:
        infer_bbox_tmp = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
        for label_bbox in label_bboxs:
            label_bbox_tmp = BBox(
                label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3])
            if iou_calculate(label_bbox_tmp, infer_bbox_tmp) >= thres:
                count += 1
                break
    return count

def read_json_filter_label_infer_bboxs(label_filename, infer_filename):
    f = open(label_filename, encoding='utf-8')
    file = json.load(f)
    f.close()
    objs = file["objects"]
    width = file["width"]
    height = file["height"]
    label_bboxs = []
    label_bboxs_temp = list()
    class_name_bboxs = []
    pose_orientation_bboxs = []
    toward_orientation_bboxs = []
    characteristic_bboxs = []
    detect_filter_bboxs = []
    truncation_bboxs = []
    for obj in objs:
        label_bbox_temp = obj["bbox"]
        if obj["class_name"] != "traffic_light":
            class_name_bboxs.append(label_bbox_temp)
            continue
        if obj['pose_orientation'] not in [0, 1]:
            # 非横向和竖向的灯滤掉
            pose_orientation_bboxs.append(label_bbox_temp)
            continue
        if obj['toward_orientation'] not in [0]:  # V18背面标的是1, V21背面标的是2
            #非正对的灯滤掉
            toward_orientation_bboxs.append(label_bbox_temp)
            continue
        if obj['characteristic'] not in [0, 1]:
            #非通行灯和行人灯滤掉
            characteristic_bboxs.append(label_bbox_temp)
            continue
        if math.ceil(obj['bbox'][2]) < traffic_light_detect_filter_bbox or math.ceil(obj['bbox'][3]) < traffic_light_detect_filter_bbox:
            # 任意一边小于10滤掉
            detect_filter_bboxs.append(label_bbox_temp)
            continue
        if obj['truncation'] == 1:
            #截断的灯滤掉
            truncation_bboxs.append(label_bbox_temp)
            continue

        label_bboxs_temp.append(label_bbox_temp)

    # 过滤后还有要对比的bbox
    if len(label_bboxs_temp) != 0:
        # array_temp = np.array(tuple(label_bboxs_temp))
        # label_bboxs.append(array_temp)
        label_bboxs.append(label_bboxs_temp)

    # 如果当前图片所有信息都是被过滤掉的(label_bboxs=0)，则对应的检测结果全部过滤掉
    if filter_infer_txt == True and len(label_bboxs) == 0:
        # print("label_filename : %s" % label_filename)
        infer_bboxs = []
        return label_bboxs, infer_bboxs

    infer_bboxs = []
    infer_bboxs_temp = list()
    with open(infer_filename, 'r') as fp:
        # print("infer_filename:%s" % infer_filename)
        lines = fp.readlines()
        # print("lines:%d" % len(lines))
        trafficlight_count = 0
        trafficliht_step = 8
        trafficlight_useless = 3
        for line in lines:
            if "trafficlight total:" in line:
                trafficlight_count = int(line[19:-1])
                # print("trafficlight_count:%s" % trafficlight_count)

        for i in range(trafficlight_count):
            step_tmp = i * trafficliht_step + trafficlight_useless
            trafficlight_number = int(lines[step_tmp][20:-1])
            # print("trafficlight_number:%s" % trafficlight_number)

            if trafficlight_number != i:
                print(f"trafficlight_number : {trafficlight_number} is not exist!")
                exit(0)

            score = float(lines[step_tmp + 1][6:-1])
            # print("score:%s" % score)
            x = int(lines[step_tmp + 2][2:-1])
            # print("x:%s" % x)
            y = int(lines[step_tmp + 3][2:-1])
            # print("y:%s" % y)
            width = int(lines[step_tmp + 4][6:-1])
            # print("width:%s" % width)
            height = int(lines[step_tmp + 5][7:-1])
            # print("height:%s\n" % height)
            infer_bbox_temp = [x, y, width, height, score]

            if filter_infer_bboxs == True:
                class_name_bboxs_flag = False
                for bbox in class_name_bboxs:
                    if iou_bbox(bbox, infer_bbox_temp, iou_debug_thres) == True:
                        class_name_bboxs_flag = True
                        break
                if class_name_bboxs_flag == True:
                    continue

                pose_orientation_bboxs_flag = False
                for bbox in pose_orientation_bboxs:
                    if iou_bbox(bbox, infer_bbox_temp, iou_debug_thres) == True:
                        pose_orientation_bboxs_flag = True
                        break
                if pose_orientation_bboxs_flag == True:
                    continue

                toward_orientation_bboxs_flag = False
                for bbox in toward_orientation_bboxs:
                    if iou_bbox(bbox, infer_bbox_temp, iou_debug_thres) == True:
                        toward_orientation_bboxs_flag = True
                        break
                if toward_orientation_bboxs_flag == True:
                    continue

                characteristic_bboxs_flag = False
                for bbox in characteristic_bboxs:
                    if iou_bbox(bbox, infer_bbox_temp, iou_debug_thres) == True:
                        characteristic_bboxs_flag = True
                        break
                if characteristic_bboxs_flag == True:
                    continue

                detect_filter_bboxs_flag = False
                for bbox in detect_filter_bboxs:
                    if iou_bbox(bbox, infer_bbox_temp, iou_debug_thres) == True:
                        detect_filter_bboxs_flag = True
                        break
                if detect_filter_bboxs_flag == True:
                    continue

                truncation_bboxs_flag = False
                for bbox in truncation_bboxs:
                    if iou_bbox(bbox, infer_bbox_temp, iou_debug_thres) == True:
                        truncation_bboxs_flag = True
                        break
                if truncation_bboxs_flag == True:
                    continue

                truncation_bboxs_flag = False
                for bbox in truncation_bboxs:
                    if iou_bbox(bbox, infer_bbox_temp, iou_debug_thres) == True:
                        truncation_bboxs_flag = True
                        break
                if truncation_bboxs_flag == True:
                    continue

            infer_bboxs_temp.append(infer_bbox_temp)

        # 只要有标注信息，不管有没有检测出东西来都要添加并返回
        # array_temp = np.array(tuple(infer_bboxs_temp))
        # infer_bboxs.append(array_temp)
        infer_bboxs.append(infer_bboxs_temp)

    return label_bboxs, infer_bboxs

# COCO test
def infer_results(label_path, infer_path):
    print("label_path : %s" % label_path)
    print("infer_path : %s" % infer_path)
    label_suffix = 'json'
    infer_suffix = 'txt'
    label_files = get_files(label_path, label_suffix)
    infer_files = get_files(infer_path, infer_suffix)
    label_path = str(Path(label_path).resolve())
    label_json_count = 0
    label_bboxs_count = 0
    infer_txt_count = 0
    infer_bboxs_count = 0
    match_bboxs_count = 0
    infer_bboxs_results = list()
    for infer_file in infer_files:
        file_basename = os.path.basename(infer_file)
        file_prefix = os.path.splitext(file_basename)[0]
        index = file_prefix.rfind("{a}_".format(a=debug_os))
        label_pre = file_prefix[0: index]
        label_suf = file_prefix[index + 4:]
        label_file = str(label_path) + '/' + label_pre + \
            label_suf + '.' + label_suffix
        if not label_file in label_files:
            print(f"{label_file} is not exist!")
            exit(0)
        label_bboxs, infer_bboxs = read_json_filter_label_infer_bboxs(
            label_file, infer_file)
        if len(label_bboxs) != 0:
            label_json_count += 1
            label_bboxs_count += len(label_bboxs[0])

        if len(infer_bboxs) != 0:
            infer_txt_count += 1
            infer_bboxs_count += len(infer_bboxs[0])
            infer_bboxs_results.append(infer_bboxs)

        if calculate_method == "iou":
            match_bboxs_count += iou_bboxs(label_bboxs,
                                           infer_bboxs, iou_debug_thres)

    print("infer_bboxs_results label_json_count  : %d" % label_json_count)
    print("infer_bboxs_results label_bboxs_count : %d" % label_bboxs_count)
    print("infer_bboxs_results infer_txt_count   : %d" % infer_txt_count)
    print("infer_bboxs_results infer_bboxs_count : %d" % infer_bboxs_count)
    print("infer_bboxs_results len               : %d" % len(infer_bboxs_results))

    if calculate_method == "iou":
        TP_TRD = float(match_bboxs_count)
        FN_TRD = float(label_bboxs_count) - TP_TRD
        FP_TRD = float(infer_bboxs_count) - TP_TRD
        TN_TRD = 0.0
        accuracy_rate = round(
            ((TP_TRD + TN_TRD) / (TP_TRD + FN_TRD + FP_TRD + TN_TRD)) * 100, 1)
        precision_rate = round((TP_TRD / (TP_TRD + FP_TRD)) * 100, 1)
        recall_rate = round((TP_TRD / (TP_TRD + FN_TRD)) * 100, 1)
        print(f"------label bbox num:\t{label_bboxs_count} ------")
        print(f"------infer bbox num:\t{infer_bboxs_count} ------")
        print(f"------match bbox num:\t{match_bboxs_count} ------")
        print(f"------accuracy rate:\t{accuracy_rate}% ------")
        print(f"------precision rate:\t{precision_rate}% ------")
        print(f"------recall rate:\t{recall_rate}% ------")

    return infer_bboxs_results
