import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from CustomEval import CustomEval
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from traffic_light_dataset import *


pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# 1
#initialize COCO ground truth api
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)

test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(1920, 1080),
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			# dict(type='LocationEmbedding'),
			# dict(type='NormalizeWithLocation', **img_norm_cfg),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
															'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
		])
]

# 根据实际情况进行配置
V21_2022_Q1_hd_QA_fz = [
	'/ssd1/fengzhen_ssd/data_set/',
	['Featured_HD_2M_1000_data_set']
]

# 测试类型，根据代码猜测使用此值
mode = "QA"
# 管道
pipeline = test_pipeline
# 工作路径
work_dir = '../../data/COCO_output'
# 测试的图像和标签，以卡片形式存在
card_group = [V21_2022_Q1_hd_QA_fz]
# 项目名称
project = 'hd'
# 负样本比例
neg_sample_ratio = 0.0
# 作物大小
crop_size = (512, 512)
# 用于保存检测错误的样本
view_eval_result = False
check_data_infos = False

# work_dir = '../../data/'
# data_root = '../../data/'
# dataset_dir = '../../data/QA/'
# card_id = "HD_2M_0_1999"
# json_name = hd + "_" + "QA" + ".json" = 'hd_QA.json'
# 将标注文件所有内容写入到这个地址的json中
# ann_file = os.path.join(self.dataset_dir, self.json_name)
# card_id_coco_format_root = '../../data/QA/' + 'HD_2M_0_1999/'

print("First step")
print("----------------------------------------------")
dataset = TrafficLightDetDataset(mode, pipeline, work_dir, card_group,
									project, neg_sample_ratio, crop_size, view_eval_result, check_data_infos)
print("----------------------------------------------")

CLASSES = ('traffic_light')

gt_path = work_dir + "/QA/hd_QA.json"
det_path = "/home/fengzhen/fengzhen_ssd/data_set_python_script/trafficlight_check_script/data/精选1000帧新数据集_胡佳纯输出/onnx_detection_result/" + "/"

with open(gt_path) as fr:
	result = json.load(fr)

dict_imgname_id = {}
for i in range(len(result['images'])):
	img_name = result['images'][i]['file_name'].split('/')[-1]
	img_id = result['images'][i]['id']
	dict_imgname_id[img_name] = img_id


det_json_list = os.listdir(det_path)
det_bbox_list = []
for j in range(len(det_json_list)):
	det_img_id = dict_imgname_id[det_json_list[j][:-5] + '.jpg']

	with open(det_path + det_json_list[j]) as fr_det:
		result_det = json.load(fr_det)
	for k in range(len(result_det)):
		det_result_one_bbox = result_det[k]
		det_result_one_bbox['image_id'] = det_img_id
		det_result_one_bbox['category_id'] = 1
		det_bbox_list.append(det_result_one_bbox)
print("len det_bbox_list ", len(det_bbox_list))


cocoGt = COCO(gt_path)
cocoDt=cocoGt.loadRes(det_bbox_list)
cocoEval = CustomEval(cocoGt, cocoDt, 'bbox')

# cocoEval.params.catIds = [1]
# cocoEval.params.imgIds = [iter0 for iter0 in range(1000)]
cocoEval.params.maxDets = [100,300,1000]

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
