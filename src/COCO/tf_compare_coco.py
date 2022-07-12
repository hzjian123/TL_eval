import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
from .traffic_light_dataset import *
from .CustomEval import *

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

if __name__ == "__main__":
    # 1
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    print('Running demo for *%s* results.' % (annType))

    # 2
    #initialize COCO ground truth api
    dataDir = '../'
    dataType = 'val2014'
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
    cocoGt = COCO(annFile)
    #loading annotations into memory...
    #Done(t=8.01s)
    #creating index...
    #index created!

    # 3
    #initialize COCO detections api
    resFile = '%s/results/%s_%s_fake%s100_results.json'
    resFile = resFile % (dataDir, prefix, dataType, annType)
    cocoDt = cocoGt.loadRes(resFile)
    #Loading and preparing results...
    #DONE(t=0.05s)
    #creating index...
    #index created!

    # 4
    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]

    # 5
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
