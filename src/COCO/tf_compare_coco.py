import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
from traffic_light_dataset import *
from tf_compare_results import *

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

if __name__ == "__main__":
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
    dataset = TrafficLightDetDataset(mode, pipeline, work_dir, card_group, project, neg_sample_ratio, crop_size, view_eval_result, check_data_infos)
    print("----------------------------------------------")

    # 2
    # running evaluation
    input_lable_path = '/home/fengzhen/fengzhen_ssd/data_set/Featured_HD_2M_1000_data_set/labels'

    input_infer_path = "/home/fengzhen/fengzhen_ssd/data_set_python_script/trafficlight_check_script/data/2022-07-27更换1000帧精选新数据集_冯震输出结果"

    print("\nSecond step")
    print("----------------------------------------------")
    print(dataset.evaluate(infer_results(input_lable_path, input_infer_path)))
    print("----------------------------------------------")
    # 测试准召率，不走coco后续流程
    # infer_results(input_lable_path, input_infer_path)
    # 打印检测结果转换后的内容
    # print(infer_results(input_lable_path, input_infer_path))
