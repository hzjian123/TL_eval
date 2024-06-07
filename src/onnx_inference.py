import onnxruntime as rt
import cv2
import math
import numpy
from mmdet.core.post_processing import multiclass_nms
import os

import numpy as np
import onnx
import torch
import json
from pathlib import Path
import copy


TARGET_MIN_SIZE = 32
TARGET_INPUT_SIZE = 224

def calculate_boundary_of_the_sub_img(gt_bbox, img_width, img_height):
    """
    :param gt_bbox:             x,y,w,h
    :param img_width:
    :param img_height:
    :param target_min_size:
    :param target_bbox_size:
    :return:                    x1, y1, x2, y2
    """
    scale = TARGET_MIN_SIZE / min(gt_bbox[2], gt_bbox[3])
    bbox_size = TARGET_INPUT_SIZE / scale
    x1 = max(0, gt_bbox[0] + gt_bbox[2] / 2 - bbox_size / 2)
    y1 = max(0, gt_bbox[1] + gt_bbox[3] / 2 - bbox_size / 2)
    x2 = min(gt_bbox[0] + gt_bbox[2] / 2 + bbox_size / 2, img_width)
    y2 = min(gt_bbox[1] + gt_bbox[3] / 2 + bbox_size / 2, img_height)
    return [int(x1), int(y1), int(x2), int(y2)], scale


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
def generate_point_coordinates(strides, feature_map_sizes):
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
    # print(image.ndim)
    # print(image.shape[-1])
    # input_shape = [1440, 2560, 3]  # 网络输入图像的 宽 高 通道
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

    # res = sess.run([proposal_bbox, proposal_score], {input: input_batch.astype(np.float32)})
    proposal_bbox, proposal_score = sess.run(None, {input: input_batch.astype(np.float32)})

    # out = np.array(res)
    # print("out, ", out)
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




def predclas(sess, img):
    input_name = sess.get_inputs()[0].name
    pred_shape = sess.get_outputs()[0].name
    pred_color = sess.get_outputs()[1].name
    pred_toward = sess.get_outputs()[2].name
    img_roi = copy.deepcopy(img)
    longer_side = max(img_roi.shape[0], img_roi.shape[1])
    resize_factor = 128.0 / longer_side
    resize_height = int(np.round(img_roi.shape[0] * resize_factor))
    resize_width = int(np.round(img_roi.shape[1] * resize_factor))
    # print("resize_height: ", resize_height, "resize_width: ", resize_width)
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



# def predclas(cls_sess, img, x1,y1,x2,y2):
#     input_name = cls_sess.get_inputs()[0].name
#     pred_color = cls_sess.get_outputs()[0].name
#     pred_shape = cls_sess.get_outputs()[1].name
#     pred_toward = cls_sess.get_outputs()[2].name

#     sub_img_box, scale = calculate_boundary_of_the_sub_img([x1, y1, x2 - x1, y2 - y1], img.shape[1], img.shape[0])
#     sub_img = copy.deepcopy(img[sub_img_box[1]:sub_img_box[3], sub_img_box[0]:sub_img_box[2], :])
    
#     sub_img = cv2.resize(sub_img, (128, 128), interpolation=cv2.INTER_LINEAR)
#     # cv2.imwrite(f"/mnt/ve_share/hjc/tl_det/{str(x1)}_{str(y1)}.jpg", sub_img)
#     sub_img = sub_img[None, ...]
#     sub_img = sub_img[:, :, :, [2, 1, 0]]
#     sub_img = sub_img.transpose(0, 3, 1, 2)
#     sub_img = (sub_img - np.array([0, 0, 0], dtype=np.float)[None, :, None, None]) / np.array(
#         [255., 255., 255.], dtype=np.float)[None, :, None, None]

#     pred_color, pred_shape, pred_toward = cls_sess.run(
#         None, {input_name: sub_img.astype(np.float32)})
#     result = {}
#     print(pred_color[0].shape)
#     print(pred_shape[0].shape)
#     print(pred_toward[0].shape)
#     result['color_score'] = pred_color[0]
#     result['shape_score'] = pred_shape[0]
#     result['toward_score'] = pred_toward[0]
#     result['color_label'] = np.argmax(pred_color[0])
#     result['shape_label'] = np.argmax(pred_shape[0])
#     result['toward_label'] = np.argmax(pred_toward[0])

#     for key in result.keys():
#         if type(result[key]) == numpy.int64:
#             result[key] = int(result[key])
#         else:
#             result[key] = list(result[key])
#             result[key] = list(map(str, result[key]))
#         # print(key," is ",result[key])
#     return result


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
    'other',
    'unknow'

]

TOWARD_CLASSES = [
    'front',
    'side',
    'backside',
    'unknow'
]


onnx_file = "/mnt/ve_share/lijixiang/hzj/traffic/work-dir/trafficlight/detect_es11_daytime/det_xmt_2_v1_update_2022_08_25.onnx"
#onnx_file = "/mnt/ve_share/lijixiang/hzj/traffic/mm/work-dir/stage1_try.onnx"
cls_onnx_file = "/mnt/ve_share/lijixiang/hzj/traffic/work-dir/trafficlight/class_es11_daytime/cla_xmt_2_20230912.onnx"
input_folder = '/mnt/ve_share/lijixiang/hzj/traffic/stage1/saves/dataset_test/small_test'
save_path = Path("saves")
save_json_path = Path("saves/json_gt")
save_crop_path = Path("saves/crop")
if not save_path.exists():
    save_path.mkdir(parents=True, exist_ok=True)
if not save_crop_path.exists():
    save_crop_path.mkdir(parents=True, exist_ok=True)
if not save_json_path.exists():
    save_json_path.mkdir(parents=True, exist_ok=True)

save_json = True

if __name__ == "__main__":
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx_path = onnx_file
    sess = rt.InferenceSession(onnx_path, providers=EP_list)
    sessclas = rt.InferenceSession(cls_onnx_file, providers=EP_list)
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
        print("dealing ", index1, " ", image_list[index1], "...")
        input_path = os.path.join(input_folder, 'images',image_list[index1])
        im = cv2.imread(input_path)
        bboxes_onnx = predict_onnx(im,
                         sess,
                         onnx_input_shape,
                         number_of_point,
                         point_coordinates,
                         expanded_regression_ranges_max,
                         location_embedding,
                         classification_threshold=0.35,
                         nms_threshold=0.4,
                         class_agnostic=False)
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
            # onnx input_shape 1440 2560 3
            width_scale = im.shape[1] / onnx_input_shape[1]
            height_scale = im.shape[0]/ onnx_input_shape[0]

            for index4 in range(len(bboxes_onnx)):
                bboxes_onnx_list = bboxes_onnx.tolist()
                # cv2.rectangle(im, (int(bboxes_onnx_list[index4][0] * width_scale),
                #                    int(bboxes_onnx_list[index4][1] * height_scale)),
                #                   (int(bboxes_onnx_list[index4][2] * width_scale),
                #                    int(bboxes_onnx_list[index4][3] * height_scale)),
                #                   (0, 0, 255), 1)
                # cv2.putText(im, str(bboxes_onnx_list[index4][4])[:5],
                #                 (int(bboxes_onnx_list[index4][0] * width_scale),
                #                    int(bboxes_onnx_list[index4][1] * height_scale - 10)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                cropimg = im[int(bboxes_onnx_list[index4][1] * height_scale):int(bboxes_onnx_list[index4][3] * height_scale),
                          int(bboxes_onnx_list[index4][0] * width_scale):int(bboxes_onnx_list[index4][2] * width_scale)]

                # cv2.imwrite(os.path.join(save_crop_path, image_list[index1][:-4] + '_' + str(index4) + '.jpg'), cropimg)
                if cropimg is not None and 0 not in cropimg.shape:
                    # result = predclas(sessclas,im, int(bboxes_onnx_list[index4][0] * width_scale),
                    #                   int(bboxes_onnx_list[index4][1] * height_scale),
                    #                   int(bboxes_onnx_list[index4][2] * width_scale),
                    #                   int(bboxes_onnx_list[index4][3] * height_scale))
                    result = predclas(sessclas, cropimg)
                    # print("result['toward_label'] ", result['toward_label'])
                    # print("result['toward_score'] ", result['toward_score'])
                    # print("result['color_label'] ", result['color_label'])
                    # print("result['color_score'] ", result['color_score'])
                    # print("result['shape_label'] ", result['shape_label'])
                    # print("result['shape_score'] ", result['shape_score'])
                    cv2.putText(im, str(bboxes_onnx_list[index4][4])[:5],
                                    (int(bboxes_onnx_list[index4][0] * width_scale),
                                       int(bboxes_onnx_list[index4][1] * height_scale - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                    cv2.putText(im, COLOR_CLASSES[result['color_label']],
                                (int(bboxes_onnx_list[index4][0] * width_scale),
                                int(bboxes_onnx_list[index4][3] * height_scale + 10))
                                , cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                (108, 76, 255))
                    # cv2.putText(im, SHAPE_CLASSES[result['shape_label']],
                    #             (int(bboxes_onnx_list[index4][0] * width_scale),
                    #             int(bboxes_onnx_list[index4][3] * height_scale + 30)),
                    #             cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    #             (108, 76, 255))
                    # cv2.putText(im,TOWARD_CLASSES[result['toward_label']],(int(bboxes_onnx_list[index4][0] * width_scale),
                    #             int(bboxes_onnx_list[index4][3] * height_scale + 50)),
                    #             cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    #             (108, 76, 255))
                    det_dict = {
                        'bbox_id':index4,
                        'bbox':[bboxes_onnx_list[index4][0] * width_scale,
                                bboxes_onnx_list[index4][1] * height_scale,
                                (bboxes_onnx_list[index4][2] - bboxes_onnx_list[index4][0])*width_scale,
                                (bboxes_onnx_list[index4][3] - bboxes_onnx_list[index4][1])*height_scale],
                        'score':bboxes_onnx_list[index4][4],
                        'color':COLOR_CLASSES[result['color_label']],
                    }
                else:
                    det_dict = {
                        'bbox_id':index4,
                        'bbox':[bboxes_onnx_list[index4][0] * width_scale,
                                bboxes_onnx_list[index4][1] * height_scale,
                                (bboxes_onnx_list[index4][2] - bboxes_onnx_list[index4][0])*width_scale,
                                (bboxes_onnx_list[index4][3] - bboxes_onnx_list[index4][1])*height_scale],
                        'score':bboxes_onnx_list[index4][4],
                        'color':COLOR_CLASSES[-1],
                    }
                cv2.rectangle(im, (int(bboxes_onnx_list[index4][0] * width_scale),
                                   int(bboxes_onnx_list[index4][1] * height_scale)),
                                  (int(bboxes_onnx_list[index4][2] * width_scale),
                                   int(bboxes_onnx_list[index4][3] * height_scale)),
                                  (0, 0, 255), 1)
                det_list.append(det_dict)
            cv2.imwrite(os.path.join(save_path, 'imgs',image_list[index1]), im)


            if save_json:
                with open(os.path.join(save_json_path, image_list[index1][:-4] + '.json'), "w") as f:
                    json.dump(det_list, f, indent=4)
                print("generate onnx detection output...")
