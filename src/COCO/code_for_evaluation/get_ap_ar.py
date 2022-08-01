import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from CustomEval import CustomEval
import numpy as np

CLASSES = ( 'traffic_light')

gt_path = "/home/haomo/haomo_project/0801/hd_QA.json"
det_path = "/home/haomo/haomo_project/0801/onnx_detection_result/"

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
