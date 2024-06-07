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
import json
from pathlib import Path
import copy

from Projects.traffic_light.predict_onnx_stage_two_xmt import get_sess_onnx as get_det_sess
from Projects.traffic_light.predict_onnx_stage_two_xmt import predict_stage_two as det_predict
from Projects.traffic_light.predict_onnx_stage_two_xmt import point_coordinates as det_point_coordinates
from Projects.traffic_light.predict_onnx_stage_two_xmt import expanded_regression_ranges_max as det_expanded_regression_ranges_max
from Projects.traffic_light.predict_onnx_stage_two_xmt import stage_two_post_process


def generate_head_feature_map_size(input_shape):
    head_feature_map_sizes = [(math.ceil(input_shape[0] / 4 - 2), math.ceil(input_shape[1] / 4 - 2)),
     (math.ceil(input_shape[0] / 8 - 2), math.ceil(input_shape[1] / 8 - 2)),
     (math.ceil(input_shape[0] / 16 - 2), math.ceil(input_shape[1] / 16 - 2 )),
     (math.ceil(input_shape[0] / 32 - 2), math.ceil(input_shape[1] / 32 - 2))] #,
     # (math.ceil(input_shape[0] / 64), math.ceil(input_shape[1] / 64))]
    return head_feature_map_sizes

def generate_number_of_point(head_feature_map_sizes):
    number_of_point = [a * b for a, b in head_feature_map_sizes]
    number_of_point = sum(number_of_point)
    return number_of_point

## 计算需要的点坐标参数
def generate_point_coordinates('stride's, feature_map_sizes):
    assert len(strides) == len(feature_map_sizes)
    point_coordinates_list = []
    for stride, feature_map_size in zip(strides, feature_map_sizes):
        map_height, map_width = feature_map_size
        x_coordinates = numpy.arange(0, map_width * stride, stride)
        y_coordinates = numpy.arange(0, map_height * stride, stride)
        x_mesh, y_mesh = numpy.meshgrid(x_coordinates, y_coordinates)
        point_coordinates_list.append(numpy.concatenate([x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)], axis=-1))

    func_point_coordinates = numpy.concatenate(point_coordinates_list, axis=0)
    return func_point_coordinates


## 计算range参数
def generate_expanded_regression_ranges(regression_ranges, feature_map_sizes):
    assert len(regression_ranges) == len(feature_map_sizes)
    expanded_regression_ranges_list = []
    for regression_range, feature_map_size in zip(regression_ranges, feature_map_sizes):
        expanded_regression_ranges_list.append(
            numpy.tile(numpy.array(regression_range).reshape(1, -1), [feature_map_size[0] * feature_map_size[1], 1])
        )
    expanded_regression_ranges = numpy.concatenate(expanded_regression_ranges_list, axis=0)
    func_expanded_regression_ranges_max = expanded_regression_ranges.max(axis=1)
    return func_expanded_regression_ranges_max


## 推理结果转换维bbox
def distance2bbox(points, predicted_distances, max_shape=None):
    x1 = points[:, 0] - predicted_distances[:, 0]
    y1 = points[:, 1] - predicted_distances[:, 1]
    x2 = points[:, 0] + predicted_distances[:, 2]
    y2 = points[:, 1] + predicted_distances[:, 3]
    if max_shape is not None:
        x1 = x1.clip(0, max_shape[1])
        y1 = y1.clip(0, max_shape[0])
        x2 = x2.clip(0, max_shape[1])
        y2 = y2.clip(0, max_shape[0])
    return numpy.stack([x1, y1, x2, y2], -1)


def generate_location_embedding(input_shape):
    grad_x = np.arange(input_shape[1], dtype=np.uint8)
    grad_y = np.arange(input_shape[0], dtype=np.uint8)
    [x, y] = np.meshgrid(grad_x, grad_y)
    x_location_normed = x / np.float32(input_shape[1])
    y_location_normed = y / np.float32(input_shape[0])
    location_embedding = np.concatenate((x_location_normed[..., None], y_location_normed[..., None]), axis=2)

    return location_embedding


def pre_compute(input_shape, head_strides, head_regression_ranges):
    # 该参数可以提前计算
    head_feature_map_sizes = generate_head_feature_map_size(input_shape)
    number_of_point = generate_number_of_point(head_feature_map_sizes)
    point_coordinates = generate_point_coordinates(head_strides, head_feature_map_sizes)
    expanded_regression_ranges_max = generate_expanded_regression_ranges(head_regression_ranges, head_feature_map_sizes)
    location_embedding = generate_location_embedding(input_shape)
    return number_of_point, point_coordinates, expanded_regression_ranges_max, location_embedding


def predict_onnx(image,
            sess,
            input_shape,
            number_of_point,
            concat_point_coordinates,
            expanded_regression_ranges_max,
            location_embedding,
            classification_threshold,
            nms_threshold,
            class_agnostic=False):
    assert isinstance(image, numpy.ndarray) and image.ndim == 3 and image.shape[-1] == 3
    input = sess.get_inputs()[0].name
    proposal_bbox = sess.get_outputs()[0].name
    proposal_score = sess.get_outputs()[1].name
    image = cv2.resize(image.astype(numpy.float32), (input_shape[1], input_shape[0]))
    # 归一化
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    image = image / 255.0
    # image = np.concatenate((image, location_embedding), axis=2)
    # 构造batch
    input_batch = image.transpose([2, 0, 1])[None]
    proposal_bbox, proposal_score = sess.run(None, {input: input_batch.astype(np.float32)})
    predicted_regression = proposal_bbox.reshape(number_of_point, 4)
    predicted_classification = proposal_score.reshape(number_of_point, 1)

    for i in range(number_of_point):
        predicted_classification[i][0] = 1.0 / (1.0 + math.exp(-predicted_classification[i][0]))
        for j in range(4):
            predicted_regression[i][j] = 1.0 / (1.0 + math.exp(-predicted_regression[i][j]))

    max_scores = predicted_classification.max(axis=1)
    selected_indexes = numpy.where(max_scores > classification_threshold)[0]
    if selected_indexes.size == 0:
        return []

    max_scores = max_scores[selected_indexes]
    # print(max_scores)

    predicted_classification = predicted_classification[selected_indexes]
    predicted_regression = predicted_regression[selected_indexes]
    concat_point_coordinates = concat_point_coordinates[selected_indexes]
    concat_regression_ranges_max = expanded_regression_ranges_max[selected_indexes]

    #  计算预测出来的所有bbox（ x1 y1 x2 y2）
    predicted_regression = predicted_regression * concat_regression_ranges_max[..., None]
    predicted_bboxes = distance2bbox(concat_point_coordinates,
                                     predicted_regression,
                                     max_shape=(input_shape[0], input_shape[1]))

    #    # 以下为NMS的步骤（依赖的第三方库），可以不用过于参考-------------------------------------------------------
    bg_label_padding = numpy.zeros((predicted_classification.shape[0], 1))
    predicted_classification = numpy.concatenate([predicted_classification, bg_label_padding], axis=1)

    nms_cfg = {'iou_threshold': nms_threshold,
               'class_agnostic': class_agnostic}

    nms_bboxes, nms_labels = multiclass_nms(
        multi_bboxes=torch.from_numpy(predicted_bboxes.astype(numpy.float32)),
        multi_scores=torch.from_numpy(predicted_classification.astype(numpy.float32)),
        score_thr=classification_threshold,
        nms_cfg=nms_cfg,
        max_num=-1,
        score_factors=None
    )

    if nms_bboxes.size(0) == 0:
        return []
    return nms_bboxes

def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if 'Normalize' in _['type']]
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config

def calculate_boundary_of_the_sub_img(gt_bbox, img_width, img_height):
    """
    :param gt_bbox:             x,y,w,h
    :param img_width:
    :param img_height:
    :param target_min_size:
    :param target_bbox_size:
    :return:                    x1, y1, x2, y2
    """
    TARGET_MIN_SIZE = 32
    TARGET_INPUT_SIZE = 224

    #bbox宽、高里面最小值，
    #scale表示32是宽、高中短边长度的几倍
    scale = TARGET_MIN_SIZE / min(gt_bbox[2], gt_bbox[3]) # scale <= 32
    #bbox_size是在 在128基础上按照scale比例系数调整得到的，scale>1就相当于128缩小scale倍，<1就相当于放大1/scale倍
    #bbox_size = 4 * min(宽，高)，也就是bbox_size相当于将 宽高中的短边的4倍大小
    bbox_size = TARGET_INPUT_SIZE / scale  #bbox_size >=4 
    #左上顶点：x方向由宽边中点左移（bbox_size/2）= （4 * min(宽，高)）/2 = 2 * min(宽，高)，也就是左移宽高中的短边的2倍大小  
    x1 = max(0, gt_bbox[0] + gt_bbox[2] / 2 - bbox_size / 2)
    #左上顶点：y方向由高边中点上移宽高中的短边的2倍大小
    y1 = max(0, gt_bbox[1] + gt_bbox[3] / 2 - bbox_size / 2)

    #右下顶点：x方向由宽边中点右移宽高中的短边的2倍大小
    x2 = min(gt_bbox[0] + gt_bbox[2] / 2 + bbox_size / 2, img_width)
    #右下顶点：x方向由高边中点下移宽高中的短边的2倍大小
    y2 = min(gt_bbox[1] + gt_bbox[3] / 2 + bbox_size / 2, img_height)

    #宽=x2-x1的一种情况:
    #  = (gt_bbox[0] + gt_bbox[2] / 2 + bbox_size / 2) - (gt_bbox[0] + gt_bbox[2] / 2 - bbox_size / 2)
    #  =  bbox_size
    # 第2种：宽= img_width - x1
    # 第3种：宽= x2 - 0
    # 第4中：宽= img_width

    #高=y2-y1的一种情况:
    #  = (gt_bbox[1] + gt_bbox[3] / 2 + bbox_size / 2) - (gt_bbox[1] + gt_bbox[3] / 2 - bbox_size / 2)
    #  =  bbox_size
    # 第2种：高= img_height - y1
    # 第3种：高= y2 - 0
    # 第4中：高= img_height

    #返回：将原bbox扩展成正方形（中心点不变），宽、高变成原来短边的4倍(如果超出则以图像边界为界)（前提是TARGET_MIN_SIZE=32，TARGET_INPUT_SIZE=128）
    #如果是长方形，说明有某一边扩展时超出了图像边界
    return [int(x1), int(y1), int(x2), int(y2)], scale

#调用分类模型，对当前bbox进行颜色等识别
#输入为：原图像，bbox按照图像相对于模型大小的比例在宽高方向分别缩放后的新框（以左下，右上顶点表示）                  
def predclas(cls_sess, img, x1,y1,x2,y2):
    input_name = cls_sess.get_inputs()[0].name
    pred_color = cls_sess.get_outputs()[0].name
    pred_shape = cls_sess.get_outputs()[1].name
    pred_toward = cls_sess.get_outputs()[2].name
    
    #从原图像中抠出bbox子图
    #img.shape[1]为宽，img.shape[0]为高

    #将原bbox扩展成正方形或长方形（中心点不变），宽、高变成原来短边的4倍(如果超出则以图像边界为界)（前提是TARGET_MIN_SIZE=32，TARGET_INPUT_SIZE=128）
    #如果是长方形，说明有某一边扩展时超出了图像边界
    sub_img_box, scale = calculate_boundary_of_the_sub_img([x1, y1, x2 - x1, y2 - y1], img.shape[1], img.shape[0])
    #从原图中抠出扩展后的bbox子图
    sub_img = copy.deepcopy(img[sub_img_box[1]:sub_img_box[3], sub_img_box[0]:sub_img_box[2], :])
    

    max_size = max(sub_img.shape[0:2]) #抠图中的宽高中最长边

    #因为抠图的结果可能不是正方形，这里接着设置边界框，把短边扩展成和长边一样长，变成正方形
    # top, bottom, left, right
    top = int((max_size - sub_img.shape[0]) / 2) # 高
    left = int((max_size - sub_img.shape[1]) / 2) # 宽
    #
    sub_img = cv2.copyMakeBorder(sub_img, top,
                                 max_size - sub_img.shape[0] - top,
                                 left,
                                 max_size - sub_img.shape[1] - left,
                                 cv2.BORDER_CONSTANT, None, [0, 0, 0])
    cls_input_scale = max_size / 128.

    cv2.imwrite(f"/root/result_xmt_0714_crop/{str(x1)}_{str(y1)}.jpg", sub_img)

    # img_roi_2 = cv2.imread('tmp_crop.jpg')
    # longer_side = max(img_roi.shape[0], img_roi.shape[1])
    # resize_factor = 128.0 / longer_side
    # resize_height = int(np.round(img_roi.shape[0] * resize_factor))
    # resize_width = int(np.round(img_roi.shape[1] * resize_factor))
    # print("resize_height: ", resize_height, "resize_width: ", resize_width)
    # img_roi = cv2.resize(img_roi, (resize_width, resize_height), dst=None, interpolation=cv2.INTER_LINEAR)
    
    # top = int((128 - img_roi.shape[0])/2)
    # left = int((128 - img_roi.shape[1])/2)

    # img_roi = cv2.copyMakeBorder(img_roi, top, 128 - img_roi.shape[0] - top,
    #                              left, 128 - img_roi.shape[1] - left,
    #                              cv2.BORDER_CONSTANT, None, [0,0,0])
        
    sub_light_bboxes = det_predict(sub_img,
                                   cls_sess,
                                   concat_point_coordinates=det_point_coordinates,
                                   concat_regression_ranges_max=det_expanded_regression_ranges_max,
                                   classification_threshold=0,
                                   nms_threshold=0.6,)
    
    if sub_light_bboxes is None:
        return {'color_label':4,
              'shape_label':7,
              'toward_label':3}
    #所有bbox的x1,x2,y1,y2分别做如下运算 （恢复到在原始图中的坐标）
    sub_light_bboxes[:, 0] = sub_light_bboxes[:, 0] * cls_input_scale - left + sub_img_box[0] #
    sub_light_bboxes[:, 1] = sub_light_bboxes[:, 1] * cls_input_scale - top + sub_img_box[1]
    sub_light_bboxes[:, 2] = sub_light_bboxes[:, 2] * cls_input_scale - left + sub_img_box[0]
    sub_light_bboxes[:, 3] = sub_light_bboxes[:, 3] * cls_input_scale - top + sub_img_box[1]
    
    result = {'color_label':4,
              'shape_label':7,
              'toward_label':3}
    det_proposals = stage_two_post_process(sub_light_bboxes)

    if len(det_proposals[0]) > 0: 
        # 4 4 4 --> 4
        # 1 1 1 --> 1

        # 4 4 1 -->1

        # 4 1 1 --> 1
        # 4 1 2 --> 4

        # 1 4 1 --> 1
        # 1 4 2 --> 4

        # 1 4 2 1 ---> 1
        # 4 1 1 2 ---> 4
         
        # 1 4 2 1 2 --->4
        # 1 4 2 1 2 1 1--->1
        # 1 4 2 1 2 2 1--->4
        for det_proposal in det_proposals[0]:
            color = int(det_proposal[5].item())
            shape = int(det_proposal[6].item())
            toward = int(det_proposal[7].item())
            if result['color_label']==4: #如果当前是循环的第一个bbox 或者 如果之前循环遇到的都是有效的识别结果，则保存当前的识别结果
                result['color_label'] = color
                result['shape_label'] = shape
                result['toward_label'] = toward
            elif color == result['color_label']: #result存了一个有效颜色，且如果当前bbox的和result存的颜色一样
                continue
            elif color in [0,1,2]: #result存了一个有效颜色，且当前bbox的颜色为红、黄、绿中的一种, 但和result存的颜色不一样，则result存的颜色置成unknow
                
                result['color_label'] = 4 #unknow
                result['shape_label'] = 7 #others
                result['toward_label'] = 3 #unknow

    return result

    img_roi = img_roi[:, :, :, [2, 1, 0]]
    img_roi = img_roi.transpose(0, 3, 1, 2)
    img_roi = (img_roi - np.array([0, 0, 0], dtype=np.float)[None, :, None, None]) / np.array(
        [255., 255., 255.], dtype=np.float)[None, :, None, None]

    # pred_color, pred_shape, pred_toward = cls_sess.run(
    #     None, {input_name: img_roi.astype(np.float32)})
    #cv2.imwrite("/root/code_on_10/mmdetection/mmdetection_train_Q3_v2/ttttttttest_crop.jpg", img_roi)
    pred_color, pred_shape, pred_toward = cls_sess.run(
        None, {input_name: img_roi.astype(np.float32)})
    result = {}
    print(pred_color[0].shape)
    print(pred_shape[0].shape)
    print(pred_toward[0].shape)
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
    'unknow'

]

TOWARD_CLASSES = [
    'front',
    'side',
    'backside',
    'unknow'
]


onnx_file = "/mnt/ve_share/lijixiang/hzj/traffic/work-dir/mine/tmp.onnx"
input_folder = '/mnt/ve_share/lijixiang/hzj/traffic/stage1/saves/dataset_test/small_test'
save_path = Path('saves')#Path("/root/result_xmt_0714")
save_json_path = Path('saves/json_gt')#Path("/root/result_xmt_0714_json")
#save_crop_path = Path('saves_de/crop')#Path("/root/result_xmt_0714_crop")
if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)
#f not save_crop_path.exists():
#    save_crop_path.mkdir(parents=True, exist_ok=True)
if not save_json_path.exists():
    save_json_path.mkdir(parents=True, exist_ok=True)

save_json = True

if __name__ == "__main__":
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx_path = onnx_file
    sess = rt.InferenceSession(onnx_path, providers=EP_list)
    #sessclas = rt.InferenceSession(cls_onnx_file, providers=EP_list)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    onnx_input_shape = [1440, 2560, 3]
    number_of_point, point_coordinates, \
        expanded_regression_ranges_max, location_embedding = \
            pre_compute(onnx_input_shape,
                         [4, 8, 16, 32],
                         list(((0, 32), (32, 64), (64, 128), (128, 512))))

    image_list = os.listdir(input_folder+'/images')
    image_list.sort()
    

    for index1 in range(len(image_list)):
        print("dealing ", index1, " ", image_list[index1], "...",os.path.join(input_folder,'images',image_list[index1]))
        im = cv2.imread(os.path.join(input_folder,'images',image_list[index1]))

        bboxes_onnx = predict_onnx(im,
                         sess,
                         onnx_input_shape,
                         number_of_point,
                         point_coordinates,
                         expanded_regression_ranges_max,
                         location_embedding,
                         classification_threshold=0.5,
                         nms_threshold=0.4,
                         class_agnostic=True)
        print('')
        if isinstance(bboxes_onnx, list):
            if len(bboxes_onnx) == 0:
                print("no detection onnx")
            if save_json:
                det_list = []
                with open(os.path.join(save_json_path, image_list[index1][:-4] + '.json'), "w") as f:
                    json.dump(det_list, f, indent=4)
                print("generate onnx detection output...")
                # continue
        else:
            bboxes_onnx = bboxes_onnx.detach().numpy()

            det_list = []
            # im 1080*1920*3
            # onnx input_shape 1440（高） 2560（宽） 3
            width_scale = im.shape[1] / onnx_input_shape[1] #图像到检测模型输入大小的宽比例系数 
            height_scale = im.shape[0]/ onnx_input_shape[0] #图像到检测模型输入大小的高比例系数 

            for index4 in range(len(bboxes_onnx)):
                bboxes_onnx_list = bboxes_onnx.tolist()
                cropimg = im[int(bboxes_onnx_list[index4][1] * height_scale):int(bboxes_onnx_list[index4][3] * height_scale),
                          int(bboxes_onnx_list[index4][0] * width_scale):int(bboxes_onnx_list[index4][2] * width_scale)]

                # cv2.imwrite(os.path.join(save_crop_path, image_list[index1][:-4] + '_' + str(index4) + '.jpg'), cropimg)
                if cropimg is not None and 0 not in cropimg.shape: #抠图成功且子图个方向均不为0
                    det_dict = {
                        'bbox_id':index4,
                        'bbox':[bboxes_onnx_list[index4][0] * width_scale,
                                bboxes_onnx_list[index4][1] * height_scale,
                                (bboxes_onnx_list[index4][2] - bboxes_onnx_list[index4][0])*width_scale,
                                (bboxes_onnx_list[index4][3] - bboxes_onnx_list[index4][1])*height_scale],
                        'score':bboxes_onnx_list[index4][4],
                        'color':0,# Eachann
                    }
                else:
                    det_dict = {
                        'bbox_id':index4,
                        'bbox':[bboxes_onnx_list[index4][0] * width_scale,
                                bboxes_onnx_list[index4][1] * height_scale,
                                (bboxes_onnx_list[index4][2] - bboxes_onnx_list[index4][0])*width_scale,
                                (bboxes_onnx_list[index4][3] - bboxes_onnx_list[index4][1])*height_scale],
                        'score':bboxes_onnx_list[index4][4],
                        'color':0,
                    }
                cv2.rectangle(im, (int(bboxes_onnx_list[index4][0] * width_scale),
                                   int(bboxes_onnx_list[index4][1] * height_scale)),
                                  (int(bboxes_onnx_list[index4][2] * width_scale),
                                   int(bboxes_onnx_list[index4][3] * height_scale)),
                                  (0, 0, 255), 1)
                det_list.append(det_dict)
            #cv2.imwrite(os.path.join(save_path, image_list[index1]), im)


            if save_json:
                with open(os.path.join(save_json_path, image_list[index1][:-4] + '.json'), "w") as f:
                    json.dump(det_list, f, indent=4)
                print("generate onnx detection output...")
