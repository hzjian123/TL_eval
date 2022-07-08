from pathlib import Path
import os
import sys
import glob
from cv2 import copyTo
# from turtle import width
import numpy as np
import argparse
import json
import math

# data_set_name = "ICU30"
data_set_name = "HD2.0"

debug_os = "qnx"
# debug_os = "x86"

# calculate_method = "pixel"
calculate_method = "iou"

# 使用pixel方式该变起作用
# pixel_debug_thres = 2
pixel_debug_thres = 4

# 使用iou方式该变量起作用
iou_debug_thres = 0.5

input_lable_path = '../data/HD_label_0_999'
# input_lable_path = "/ssd1/fengzhen_ssd/data_set/HD_2M_15000_data_set"

input_infer_path = "../data/2022-07-02障碍物+红绿灯+点云检测集成测试/traffic_light_qnx_output"

# 过滤条件-bbox像素小于该值过滤
traffic_light_detect_filter_bbox = 4


def haomo2coco(data_root, card_id, save_dir, json_name, dataset_mode=None, haomo_annotation_parser=None, bbox_filters=[]):
    card_id_root = os.path.join(data_root, card_id)
    card_id_image_root = os.path.join(card_id_root, 'images')
    assert os.path.exists(
        card_id_image_root), 'card_id_image_root does not exist!!!'
    card_id_annotation_root = os.path.join(card_id_root, 'labels')
    assert os.path.exists(
        card_id_annotation_root), 'card_id_annotation_root does not exist!!!'
    card_id_coco_format_root = os.path.join(save_dir, card_id)
    if not os.path.exists(card_id_coco_format_root):
        os.makedirs(card_id_coco_format_root)
    if json_name in os.listdir(card_id_coco_format_root):
        print(card_id, " already converted.")
        return
    card_id_image_path_list = [os.path.join(card_id_image_root, file_name) for file_name in
                               os.listdir(card_id_image_root) if file_name.lower().endswith(image_suffixes)]
    card_id_annotation_path_list = [os.path.join(card_id_annotation_root, file_name) for file_name in
                                    os.listdir(card_id_annotation_root) if file_name.lower().endswith('json')]
    assert len(card_id_image_path_list) == len(
        card_id_annotation_path_list), card_id
    index_list = [i for i in range(len(card_id_annotation_path_list))]
    train_image_counter = 0
    train_instance_counter = 0
    train_image_format = list()
    train_annotation_format = list()
    for i, index in enumerate(index_list):
        # print("==== index ", i, ",", index)
        annotation_path = card_id_annotation_path_list[index]
        image_path = os.path.join(card_id_image_root, os.path.basename(
            annotation_path).split('.')[0] + '.jpg')
        # print("======= step1 =====")
        assert image_path in card_id_image_path_list
        if dataset_mode == 'debug' and random.random() <= 0.995:
            # print("dataset_mode ", dataset_mode)
            # print("random.random() ", random.random())
            # print("===== step2 =====")
            continue
        parse_results = haomo_annotation_parser(annotation_path, *bbox_filters)
        if parse_results is None:
            # print("====== step3 =====")
            continue
        else:
            temp_image_info_dict, temp_annotation_list = parse_results
        # 为 image 添加新的 key
        temp_image_info_dict['id'] = train_image_counter
        temp_image_info_dict['file_name'] = image_path  # 这里直接保存整个图像的绝对路径
        train_image_format.append(temp_image_info_dict)
        for temp_bbox in temp_annotation_list:
            temp_bbox['id'] = train_instance_counter
            temp_bbox['image_id'] = train_image_counter
            train_annotation_format.append(temp_bbox)
            train_instance_counter += 1
        train_image_counter += 1
    # print("====train_image_format size ", len(train_image_format))
    #  保存coco标注文件
    print("==== image size ", len(train_image_format))
    print("==== annotation size ", len(train_annotation_format))
    print(card_id, ' saving coco format...')
    train_coco_annotation_format = dict()
    train_coco_annotation_format['info'] = 'haomo'
    train_coco_annotation_format['license'] = 'None'
    train_coco_annotation_format['images'] = train_image_format
    train_coco_annotation_format['annotations'] = train_annotation_format
    train_coco_annotation_format['categories'] = [{'id': cat_id, 'name': cat_name} for cat_name, cat_id in
                                                  category_name_to_id.items()]
    with open(os.path.join(card_id_coco_format_root, json_name), 'w') as fout:
        json.dump(train_coco_annotation_format, fout, indent=4)

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


def read_json(filename):
    f = open(filename, encoding='utf-8')
    file = json.load(f)
    f.close()
    objs = file["objects"]
    width = file["width"]
    height = file["height"]
    bboxs = []
    for obj in objs:
        if obj["class_name"] != "traffic_light":
            continue
        if obj['pose_orientation'] not in [0, 1]:
            # 非横向和竖向的灯滤掉
            continue
        if obj['toward_orientation'] not in [0]:  # V18背面标的是1, V21背面标的是2
            #非正对的灯滤掉
            continue
        if obj['characteristic'] not in [0, 1]:
            #非通行灯和行人灯滤掉
            continue
        if math.ceil(obj['bbox'][2]) < traffic_light_detect_filter_bbox or math.ceil(obj['bbox'][3]) < traffic_light_detect_filter_bbox:
            # 任意一边小于10滤掉
            continue
        if obj['truncation'] == 1:
            #截断的灯滤掉
            continue
        bbox = obj["bbox"]
        # print(type(bbox))
        if data_set_name == "ICU30":
            if height == 1080:
                bbox = [round(i * 2) for i in bbox]
            else:
                bbox = [round(i) for i in bbox]
        elif data_set_name == "HD2.0":
            bbox = [round(i) for i in bbox]
        bboxs.append(bbox)
    return bboxs


def iou_bbox(label_bbox, infer_bbox, thres=0.5):
    label_bbox_tmp = BBox(
        label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3])
    infer_bbox_tmp = BBox(
        infer_bbox[0], infer_bbox[1], infer_bbox[2], infer_bbox[3])
    if iou_calculate(label_bbox_tmp, infer_bbox_tmp) >= thres:
        return True
    return False


def read_json_filter_label_infer_bboxs(filename, infer_bboxs_tmp):
    f = open(filename, encoding='utf-8')
    file = json.load(f)
    f.close()
    objs = file["objects"]
    width = file["width"]
    height = file["height"]
    label_bboxs = []
    class_name_bboxs = []
    pose_orientation_bboxs = []
    toward_orientation_bboxs = []
    characteristic_bboxs = []
    detect_filter_bboxs = []
    truncation_bboxs = []
    for obj in objs:
        label_bbox = obj["bbox"]
        # print(type(label_bbox))
        if data_set_name == "ICU30":
            if height == 1080:
                label_bbox = [round(i * 2) for i in label_bbox]
            else:
                label_bbox = [round(i) for i in label_bbox]
        elif data_set_name == "HD2.0":
            label_bbox = [round(i) for i in label_bbox]
        if obj["class_name"] != "traffic_light":
            class_name_bboxs.append(label_bbox)
            continue
        if obj['pose_orientation'] not in [0, 1]:
            # 非横向和竖向的灯滤掉
            pose_orientation_bboxs.append(label_bbox)
            continue
        if obj['toward_orientation'] not in [0]:  # V18背面标的是1, V21背面标的是2
            #非正对的灯滤掉
            toward_orientation_bboxs.append(label_bbox)
            continue
        if obj['characteristic'] not in [0, 1]:
            #非通行灯和行人灯滤掉
            characteristic_bboxs.append(label_bbox)
            continue
        if math.ceil(obj['bbox'][2]) < traffic_light_detect_filter_bbox or math.ceil(obj['bbox'][3]) < traffic_light_detect_filter_bbox:
            # 任意一边小于10滤掉
            detect_filter_bboxs.append(label_bbox)
            continue
        if obj['truncation'] == 1:
            #截断的灯滤掉
            truncation_bboxs.append(label_bbox)
            continue
        label_bboxs.append(label_bbox)

    infer_bboxs = []
    for infer_bbox in infer_bboxs_tmp:
        class_name_bboxs_flag = False
        for bbox in class_name_bboxs:
            if iou_bbox(bbox, infer_bbox) == True:
                class_name_bboxs_flag = True
                break
        if class_name_bboxs_flag == True:
            continue

        pose_orientation_bboxs_flag = False
        for bbox in pose_orientation_bboxs:
            if iou_bbox(bbox, infer_bbox) == True:
                pose_orientation_bboxs_flag = True
                break
        if pose_orientation_bboxs_flag == True:
            continue

        toward_orientation_bboxs_flag = False
        for bbox in toward_orientation_bboxs:
            if iou_bbox(bbox, infer_bbox) == True:
                toward_orientation_bboxs_flag = True
                break
        if toward_orientation_bboxs_flag == True:
            continue

        characteristic_bboxs_flag = False
        for bbox in characteristic_bboxs:
            if iou_bbox(bbox, infer_bbox) == True:
                characteristic_bboxs_flag = True
                break
        if characteristic_bboxs_flag == True:
            continue

        detect_filter_bboxs_flag = False
        for bbox in detect_filter_bboxs:
            if iou_bbox(bbox, infer_bbox) == True:
                detect_filter_bboxs_flag = True
                break
        if detect_filter_bboxs_flag == True:
            continue

        truncation_bboxs_flag = False
        for bbox in truncation_bboxs:
            if iou_bbox(bbox, infer_bbox) == True:
                truncation_bboxs_flag = True
                break
        if truncation_bboxs_flag == True:
            continue

        infer_bboxs.append(infer_bbox)
    return label_bboxs, infer_bboxs


def read_txt(filename):
    bboxs = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        index = 0
        for line in lines:
            if "x:" in line:
                x = int(line[2:-1])
                y = int(lines[index + 1][2:-1])
                width = int(lines[index + 2][6:-1])
                height = int(lines[index + 3][7:-1])
                bboxs.append([x, y, width, height])
            index += 1
    return bboxs


def match_bboxs(label_bboxs, infer_bboxs, thres=4):
    count = 0
    for bbox in infer_bboxs:
        x = bbox[0]
        y = bbox[1]
        width = bbox[2]
        height = bbox[3]
        for label_bbox in label_bboxs:
            x_real = label_bbox[0]
            y_real = label_bbox[1]
            width_real = label_bbox[2]
            height_real = label_bbox[3]
            if abs(x_real - x) > thres:
                continue
            if abs(y_real - y) > thres:
                continue
            if abs(width_real - width) > thres:
                continue
            if abs(height_real - height) > thres:
                continue
            count += 1
    return count


def iou_bboxs(label_bboxs, infer_bboxs, thres=0.5):
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


def run(lable_path=input_lable_path, infer_path=input_infer_path):
    label_suffix = 'json'
    infer_suffix = 'txt'
    label_files = get_files(lable_path, label_suffix)
    infer_files = get_files(infer_path, infer_suffix)
    lable_path = str(Path(lable_path).resolve())
    lable_count = 0
    infer_count = 0
    match_count = 0
    for infer_file in infer_files:
        infer_bboxs_tmp = read_txt(infer_file)
        file_basename = os.path.basename(infer_file)
        file_prefix = os.path.splitext(file_basename)[0]
        index = file_prefix.rfind("{a}_".format(a=debug_os))
        lable_pre = file_prefix[0: index]
        lable_suf = file_prefix[index + 4:]
        lable_file = str(lable_path) + '/' + lable_pre + \
            lable_suf + '.' + label_suffix
        if not lable_file in label_files:
            print(f"{lable_file} is not exist!")
            exit(0)
        #label_bboxs = read_json(lable_file)
        label_bboxs, infer_bboxs = read_json_filter_label_infer_bboxs(
            lable_file, infer_bboxs_tmp)
        lable_count += len(label_bboxs)
        infer_count += len(infer_bboxs)
        # print(label_bboxs)
        if calculate_method == "pixel":
            match_count += match_bboxs(label_bboxs,
                                       infer_bboxs, pixel_debug_thres)
        elif calculate_method == "iou":
            match_count += iou_bboxs(label_bboxs, infer_bboxs, iou_debug_thres)
    TP_TRD = float(match_count)
    FN_TRD = float(lable_count) - TP_TRD
    FP_TRD = float(infer_count) - TP_TRD
    TN_TRD = 0.0
    accuracy_rate = round(
        ((TP_TRD + TN_TRD) / (TP_TRD + FN_TRD + FP_TRD + TN_TRD)) * 100, 1)
    precision_rate = round((TP_TRD / (TP_TRD + FP_TRD)) * 100, 1)
    recall_rate = round((TP_TRD / (TP_TRD + FN_TRD)) * 100, 1)
    print(f"------label bbox num:\t{lable_count} ------")
    print(f"------infer bbox num:\t{infer_count} ------")
    print(f"------match bbox num:\t{match_count} ------")
    print(f"------accuracy rate:\t{accuracy_rate}% ------")
    print(f"------precision rate:\t{precision_rate}% ------")
    print(f"------recall rate:\t{recall_rate}% ------")


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', help='an onnx model', required=True)
    parser.add_argument('--lable_path', nargs='+',
                        type=str, default=input_lable_path)
    parser.add_argument('--infer_path', nargs='+',
                        type=str, default=input_infer_path)
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
