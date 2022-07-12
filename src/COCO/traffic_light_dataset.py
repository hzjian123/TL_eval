# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import logging
import math
import os.path
import os.path as osp
import random
import json
import shutil
import tempfile
import warnings
from collections import OrderedDict

import cv2
import torch
from tqdm import tqdm
import torch.distributed as dist
import contextlib
import io

import mmcv
import pandas as pd
from mmcv.runner import get_dist_info
import numpy as np
import pathos.parallel
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from .CustomEval import CustomEval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from .bbox_filter import *

image_suffixes = ('jpg', 'jpeg', 'png', 'bmp')
category_name_to_id = {'TL': 1, 'TL_sub': 2}


@DATASETS.register_module()
class TrafficLightDetDataset(CocoDataset):
    CLASSES = ['TL']

    def __init__(self,
                 mode,
                 pipeline,
                 work_dir,
                 card_group,
                 project,
                 neg_sample_ratio=0.0,
                 crop_size=(512, 512),
                 view_eval_result=False,
                 check_data_infos=False,
                 **kwargs):
        assert project in ["icu30", "icu30_sub", "cp", "hd"]
        self.project_name = project
        self.dataset_mode = mode
        assert mode in ["train", "val", "QA", "debug", "dump", "negative"]
        self.json_name = project + "_" + mode + ".json"
        self.card_group = card_group
        self.dataset_dir = os.path.join(work_dir, mode)
        self.view_eval_result = view_eval_result

        if mode == "negative":
            assert neg_sample_ratio >= 0, "negative sample ratio should > 0"
            self.neg_sample_ratio = neg_sample_ratio
            self.crop_size = crop_size

        self.dump_result_to_json = False
        if mode == "dump":
            self.dump_result_to_json = True

        bbox_filters, haomo_annotation_parser = self._prepare_for_json()
        # 待判断是不是dist_train
        self.rank, self.world_size = get_dist_info()
        if self.rank == 0:
            if not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)
            self.generate_json(bbox_filters, haomo_annotation_parser)
        if self.world_size > 1:
            dist.barrier()
        ann_file = os.path.join(self.dataset_dir, self.json_name)

        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            classes=self.CLASSES,
            **kwargs)
        if mode in ["train", "debug"]:
            # 统计， 写进log里
            self._set_weights_and_do_statistics()
        if check_data_infos:
            if 'test_mode' in kwargs.keys() and kwargs['test_mode']:
                valid_inds = self._filter_imgs()
                # self._save_neg_img(valid_inds, os.path.join(work_dir, "check"))
                self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.rank == 0:
                self._check_data_infos(os.path.join(work_dir, "check"))
            if self.world_size > 1:
                dist.barrier()
            exit(0)

    def _prepare_for_json(self):
        # 上面的负责检查数据格式， 下面的负责组合多个标注属性
        bbox_filters = [
            bbox_filter_pose_orientation,  # 过滤没有方向字段的框与非横、竖的框
            # bbox_filter_toward_orientation,  # 过滤没有朝向字段的框与非正面侧面的框
            bbox_filter_characteristic,  # 过滤没有属性字段的框与非通行灯，非行人灯
            # bbox_filter_characteristic_cars,  # 过滤没有属性字段的框与非通行灯
            bbox_filter_toward_orientation_front,
        ]
        haomo_annotation_parser = self.parse_haomo_annotation
        if self.dataset_mode in ["train", "debug", "negative"]:
            bbox_filters.extend([
                bbox_filter_size_8,  # 过滤任意变长小于8的框
                bbox_filter_truncation,  # 过滤没有截断字段的框与被标注为截断的框
                bbox_filter_ext_occlusion,  # 过滤没有遮挡字段的框与子灯个数小于3且被标注为遮挡的框
                # bbox_filter_sublights_0,  # 通行灯 and 子灯个数0 and 非正面
                bbox_filter_ratio,  # ratio < 1.6
            ])
            if self.dataset_mode == "negative":
                haomo_annotation_parser = self.parse_negative_sample
        elif self.dataset_mode in ["QA", "val"]:
            bbox_filters.extend([
                bbox_filter_size_4,  # 过滤任意变长小于10的框
                # bbox_filter_toward_orientation,  # 过滤没有朝向字段的框与非正面侧面的框
                bbox_filter_truncation,
            ])
        elif self.dataset_mode == "dump":
            bbox_filters = [
                bbox_filter_characteristic  # 非通行灯，非人行横到灯
            ]

        return bbox_filters, haomo_annotation_parser

    def _save_neg_img(self, valid_inds, save_root):
        for idx, data_info in enumerate(self.data_infos):
            if idx not in valid_inds:
                save_dir = os.path.join(
                    save_root, data_info['filename'].split("/")[-3], "neg")
                mmcv.mkdir_or_exist(save_dir)
                shutil.copyfile(
                    data_info['filename'], save_dir + os.path.basename(data_info['filename']))

    def _check_data_infos(self, data_check_dir):
        color = (108, 76, 255)
        for data_info in tqdm(self.data_infos):
            draw_img = cv2.imread(data_info['filename'])
            img_save_root = os.path.join(
                data_check_dir, data_info['filename'].split("/")[-3], "pos")
            mmcv.mkdir_or_exist(img_save_root)
            assert draw_img is not None
            ann_ids = self.coco.getAnnIds(data_info['id'])
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                cv2.rectangle(draw_img, (int(ann["bbox"][0]), int(ann["bbox"][1])),
                              (math.ceil(ann["bbox"][2] + ann["bbox"][0]),
                               math.ceil(ann["bbox"][3] + ann["bbox"][1])),
                              color, thickness=1)
            cv2.imwrite(os.path.join(img_save_root, os.path.basename(
                data_info['filename'])), draw_img)

    def _filter_imgs(self, min_size=32):
        # TODO: 根据标注规则过滤训练图片
        valid_inds = []
        for img_info in self.data_infos:
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if len(self.coco.getAnnIds(img_info["id"])) == 0:
                if self.filter_empty_gt:
                    continue
            valid_inds.append(img_info["id"])
        return valid_inds

    def generate_json(self, bbox_filters, haomo_annotation_parser):
        all_card_list = []
        for data_root, card_id_list in self.card_group:
            for card_id in card_id_list:
                assert os.path.exists(os.path.join(data_root, card_id)), '%s does not exist!!!' % os.path.join(
                    data_root, card_id)
                # 用于保存检测错误的样本
                if self.view_eval_result:
                    if not os.path.exists(os.path.join(self.dataset_dir, card_id, "view_result")):
                        os.makedirs(os.path.join(
                            self.dataset_dir, card_id, "view_result"))
                if self.dump_result_to_json:
                    if not os.path.exists(os.path.join(self.dataset_dir, card_id, "neg_detections")):
                        os.makedirs(os.path.join(self.dataset_dir,
                                    card_id, "neg_detections"))
                self.haomo2coco(data_root, card_id, self.dataset_dir, self.json_name,
                                haomo_annotation_parser=haomo_annotation_parser,
                                bbox_filters=bbox_filters)
                all_card_list.append(card_id)
        if not os.path.exists(os.path.join(self.dataset_dir, self.json_name)):
            self.merged_coco_format(
                all_card_list, self.dataset_dir, self.json_name)
        else:
            print("train/val json already merged.")

    def _set_weights_and_do_statistics(self):
        # TODO: 黄色、黑色灯权重增加
        if self.rank == 0:
            short_edge = []
            long_edge = []
        _groups = np.zeros(len(self), dtype=np.uint8)
        for i, data_info in enumerate(self.data_infos):
            ann_ids = self.coco.getAnnIds(data_info['id'])
            if len(ann_ids) == 0:
                assert not self.filter_empty_gt, "bug."
                # 改图上所有gt都被过滤
                _groups[i] = 2
            else:
                anns = self.coco.loadAnns(ann_ids)
                # if rank == 0:
                #     print("set weights and statistics: [%d / %d]" % (i, _groups.shape[0]))
                for ann in anns:
                    # 以img为id，如果图中有横向灯flag为1（宽>=高）
                    if ann['bbox'][2] >= ann['bbox'][3]:
                        _groups[i] = 1
                    if self.rank == 0:
                        short = min(ann["bbox"][2], ann["bbox"][3])
                        long = max(ann["bbox"][2], ann["bbox"][3])
                        short_edge.append(int(short / 10.) * 10)
                        long_edge.append(int(long / 10.) * 10)
            if self.world_size > 1:
                dist.barrier()
        if self.rank == 0:
            self.short_edge_statistics = pd.value_counts(
                short_edge, normalize=True)
            self.long_edge_statistics = pd.value_counts(
                long_edge, normalize=True)
        if self.world_size > 1:
            dist.barrier()
        self._groups_size = np.bincount(_groups)
        self.weights = torch.ones(len(self))
        if self.project_name != "icu30_sub" and self._groups_size.size > 1:
            self.weights[_groups == 1] = max(2.0, float(
                self._groups_size[0] / 8.) / self._groups_size[1])
        self.weights[_groups == 2] = 0.0

    def draw_result_on_img(self, match_result, cocoDt, cocoGt):
        # 读取图片 TODO： 根据match_result获取图片名、gt，并与标注进行对比
        img_path = cocoGt.loadImgs(int(match_result['image_id']))[
            0]['file_name']
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        card_id = img_path.split("/")[-3]

        # 绘制标注结果
        color = (108, 76, 255)  # red
        gt_list = cocoGt.loadAnns(match_result['gtIds'])
        for gt_id in gt_list:
            bbox = gt_id["bbox"]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                          (math.ceil(bbox[2] + bbox[0]),
                           math.ceil(bbox[3] + bbox[1])),
                          color, 2)
            # cv2.putText(img, "gt", (int(bbox[0]), int(bbox[1]) - 20),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.8, color, thickness=2)
            cv2.putText(img, "scale: %.2f" % (max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])),
                        (int(bbox[0]), int(bbox[1]) - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, thickness=2)

        # 绘制检测结果
        color = (182, 205, 35)  # green
        dt_list = cocoDt.loadAnns(match_result['dtIds'])
        for dt_id in dt_list:
            bbox = dt_id['bbox']
            score = dt_id['score']

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                          (math.ceil(bbox[2] + bbox[0]),
                           math.ceil(bbox[3] + bbox[1])),
                          color, 2)
            cv2.putText(img, "%0.2f" % score, (int(bbox[0]), math.ceil(bbox[3] + bbox[1]) + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, thickness=2)
            cv2.putText(img, "min: %d" % math.ceil(min(bbox[2], bbox[3])),
                        (int(bbox[0]), math.ceil(bbox[3] + bbox[1]) + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, thickness=2)
            cv2.putText(img, "max: %d" % math.ceil(max(bbox[2], bbox[3])),
                        (int(bbox[0]), math.ceil(bbox[3] + bbox[1]) + 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, thickness=2)

        cv2.imwrite(os.path.join(self.dataset_dir,
                    card_id, "view_result", img_name), img)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = CustomEval(cocoGt, cocoDt, iou_type)
            # The evaluation parameters are as follows (defaults in brackets):
            #  imgIds     - [all] N img ids to use for evaluation
            #  catIds     - [all] K cat ids to use for evaluation
            #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
            #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
            #  areaRng    - [...] A=4 object area ranges for evaluation
            #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
            #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
            #  iouType replaced the now DEPRECATED useSegm parameter.
            #  useCats    - [1] if true use category labels for evaluation
            # Note: if useCats=0 category labels are ignored as in proposal scoring.
            # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            # cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.maxDets = [100, 300, 1000]
            # cocoEval.params.iouThrs = [0.5]
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()  # 计算每张图的匹配情况，保存在cocoEval.evalImgs的dtm和gtm中
                cocoEval.accumulate()

                if self.view_eval_result or self.dump_result_to_json:
                    # TODO: 过检结果 / 标注结果 成对的写入json，以训练是不是红绿灯的分类模型
                    match_results = cocoEval.evalImgs[:len(
                        cocoEval.params.imgIds)]
                    if self.dump_result_to_json:
                        eval_result_dict = dict()
                    for match_result in tqdm(match_results):
                        # 每个 match_result 代表一张图片
                        if match_result is None:
                            continue

                        if self.dump_result_to_json:
                            dump_flag = True
                            # 输出过检
                            if len(match_result['dtIds']) == 0:
                                # 无检测结果
                                dump_flag = False
                            if dump_flag:
                                _filename = self.coco.load_imgs(
                                    int(match_result['image_id']))[0]['filename']
                                eval_result_dict["card_id"] = _filename.split(
                                    "/")[-3]
                                eval_result_dict["time_stamp"] = os.path.basename(_filename).split(".")[
                                    0]
                                eval_result_dict["origin_bbox_ids"] = [gt['origin_bbox_id'] for gt in cocoEval._gts[
                                    match_result['image_id'], match_result['category_id']]]
                                eval_result_dict["neg_detections"] = []
                                # TODO: 输出未匹配上的框, 正样本从gt选取, 添加随机抖动. 由于该json要上传到git, 所以信息尽可能的精炼
                                tmp_detections = cocoEval._dts[match_result['image_id'],
                                                               match_result['category_id']]
                                for idx, det in enumerate(tmp_detections):
                                    if min(det['bbox'][2], det['bbox'][2]) < 10:
                                        continue
                                    if match_result['dtMatches'][0][idx] > 0:
                                        # 在每个阈值下的dt是否得到匹配, 0代表iou=0.5
                                        # 成功匹配到gt
                                        continue
                                    eval_result_dict["neg_detections"].append(
                                        det)

                            if len(eval_result_dict["neg_detections"]) > 0 and dump_flag:
                                with open(os.path.join(self.dataset_dir, eval_result_dict["card_id"],
                                                       "neg_detections",
                                                       eval_result_dict["time_stamp"] + ".json"), 'w') as fout:
                                    json.dump(eval_result_dict, fout, indent=4)

                        if self.view_eval_result:
                            view_flag = True
                            # 输出漏检
                            if len(match_result['gtIds']) == 0:
                                # 无gt
                                view_flag = False
                            if view_flag and min(match_result['gtMatches'][0]) > 0:
                                # 在每个阈值下的Gt是否得到匹配, 0代表iou=0.5
                                view_flag = False
                            if view_flag:
                                self.draw_result_on_img(
                                    match_result, cocoDt, cocoGt)

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'AR@100'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def parse_negative_sample(self, annotation_json_path, *bbox_filters):
        """
            制作纯负样本数据集
            :param annotation_json_path: 对应标注json文件路劲
            :param bbox_filter: bbox过滤回调函数，默认不过滤
            :return:
        """

        def add_anno_info(bbox, category):
            """

            :param bbox: [x, y, w, h]
            :return:
            """
            temp_bbox_dict = dict()
            temp_bbox_dict['category_id'] = category_name_to_id[category]
            temp_bbox_dict['bbox'] = [bbox[0], bbox[1], bbox[2], bbox[3]]
            temp_bbox_dict['area'] = (bbox[2]) * (bbox[3])
            temp_bbox_dict['segmentation'] = [
                [bbox[0],
                 bbox[1],
                 bbox[0],
                 bbox[1] + 0.5 * bbox[3],
                 bbox[0],
                 bbox[1] + bbox[3],
                 bbox[0] + 0.5 * bbox[2],
                 bbox[1] + bbox[3],
                 bbox[0] + bbox[2],
                 bbox[1] + bbox[3],
                 bbox[0] + bbox[2],
                 bbox[1] + bbox[3] - 0.5 * bbox[3],
                 bbox[0] + bbox[2],
                 bbox[1],
                 bbox[0] + bbox[2] - 0.5 * bbox[2],
                 bbox[1]]
            ]
            temp_bbox_dict['iscrowd'] = 0

            return temp_bbox_dict

        assert os.path.exists(
            annotation_json_path), 'annotation_path does not exist!!!'

        with open(annotation_json_path, 'r') as fin:
            json_content = json.load(fin)

        image_info_dict = dict()
        card_id = annotation_json_path.split("/")[-3]
        if card_id in ["61f13478761998edc227c17f", "6209fdc1d3572380d577ad42"]:
            image_info_dict['width'] = 3840
            image_info_dict['height'] = 2160
        else:
            image_info_dict['width'] = json_content['width']
            image_info_dict['height'] = json_content['height']
        image_info_dict['neg_sample'] = True

        annotation_list = list()
        assert 'objects' in json_content, 'annotation does not contain key: objects!!!'
        bboxes = json_content['objects']

        target_bboxes = []
        for bbox in bboxes:
            filter_flag = False
            for bbox_filter in bbox_filters:
                if bbox_filter(bbox):
                    filter_flag = True
                    break
            if filter_flag:
                continue
            target_bboxes.append(bbox)

        if len(target_bboxes) <= 0:
            # TODO: 如果整张图片内的gt, 都被过滤, 则整图作为一个bbox
            return None

        bboxes = np.array([anno['bbox'] for anno in target_bboxes])
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

        left = int(np.min(bboxes[:, 0]))
        right = math.ceil(np.max(bboxes[:, 2]))
        top = int(np.min(bboxes[:, 1]))
        bottom = math.ceil(np.max(bboxes[:, 3]))

        if self.project_name == "icu30_sub":
            bbox_category = "TL_sub"
        elif self.project_name == "icu30":
            bbox_category = "TL"
        else:
            raise NotImplementedError
        if left >= self.crop_size[1]:
            annotation_list.append(add_anno_info(
                [0, 0, left - 1, json_content["height"]], bbox_category))
        if top >= self.crop_size[0]:
            annotation_list.append(add_anno_info(
                [0, 0, json_content["width"], top - 1], bbox_category))
        if json_content["width"] - right >= self.crop_size[1]:
            annotation_list.append(
                add_anno_info([right + 1, 0, json_content["width"] - right - 1, json_content["height"]], bbox_category))
        if json_content["height"] - bottom >= self.crop_size[0]:
            annotation_list.append(
                add_anno_info([0, bottom + 1, json_content["width"], json_content["height"] - bottom - 1],
                              bbox_category))

        if len(annotation_list) <= 0:
            return None
        else:
            return image_info_dict, annotation_list

    def parse_haomo_annotation(self, annotation_json_path, *bbox_filters):
        """
        这个版本处理 乘用车 的标注格式
        这里需要过滤掉：
        严重遮挡的红绿灯
        :param annotation_json_path: 对应标注json文件路劲
        :param bbox_filter: bbox过滤回调函数，默认不过滤
        :return:
        """
        assert os.path.exists(
            annotation_json_path), 'annotation_path does not exist!!!'

        with open(annotation_json_path, 'r') as fin:
            json_content = json.load(fin)

        image_info_dict = dict()
        card_id = annotation_json_path.split("/")[-3]
        if card_id in ["61f13478761998edc227c17f", "6209fdc1d3572380d577ad42"]:
            image_info_dict['width'] = 3840
            image_info_dict['height'] = 2160
        else:
            image_info_dict['width'] = json_content['width']
            image_info_dict['height'] = json_content['height']
        image_info_dict['neg_sample'] = False

        annotation_list = list()
        assert 'objects' in json_content, 'annotation does not contain key: objects!!!'
        bboxes = json_content['objects']

        for bbox in bboxes:
            filter_flag = False
            for bbox_filter in bbox_filters:
                if bbox_filter(bbox):
                    filter_flag = True
                    break
            if filter_flag:
                continue

            temp_bbox_dict = dict()
            temp_bbox_dict['category_id'] = category_name_to_id['TL']
            temp_bbox_dict['bbox'] = bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2] + 1, bbox['bbox'][
                3] + 1  # x1, y1, w, h(这里我们默认w和h是包含起始点的，标注的规则里面是不包含起始点的，所以需要+1)
            temp_bbox_dict['area'] = (
                bbox['bbox'][2] + 1) * (bbox['bbox'][3] + 1)
            temp_bbox_dict['segmentation'] = [
                [bbox['bbox'][0],
                 bbox['bbox'][1],
                 bbox['bbox'][0],
                 bbox['bbox'][1] + 0.5 * bbox['bbox'][3],
                 bbox['bbox'][0],
                 bbox['bbox'][1] + bbox['bbox'][3],
                 bbox['bbox'][0] + 0.5 * bbox['bbox'][2],
                 bbox['bbox'][1] + bbox['bbox'][3],
                 bbox['bbox'][0] + bbox['bbox'][2],
                 bbox['bbox'][1] + bbox['bbox'][3],
                 bbox['bbox'][0] + bbox['bbox'][2],
                 bbox['bbox'][1] + bbox['bbox'][3] - 0.5 * bbox['bbox'][3],
                 bbox['bbox'][0] + bbox['bbox'][2],
                 bbox['bbox'][1],
                 bbox['bbox'][0] + bbox['bbox'][2] - 0.5 * bbox['bbox'][2],
                 bbox['bbox'][1]]
            ]
            temp_bbox_dict['iscrowd'] = 0
            temp_bbox_dict['origin_bbox_id'] = bbox['id']

            annotation_list.append(temp_bbox_dict)

        # 如果bbox都被过滤了，该图像不进入数据集
        # if len(annotation_list) == 0:
        #     return None
        # else:
        return image_info_dict, annotation_list

    def haomo2coco(self, data_root, card_id, save_dir, json_name, haomo_annotation_parser=None, bbox_filters=[]):
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
            annotation_path = card_id_annotation_path_list[index]
            image_path = os.path.join(card_id_image_root, os.path.basename(
                annotation_path).split('.')[0] + '.jpg')
            assert image_path in card_id_image_path_list

            if json_name == 'debug.json' and random.random() <= 0.98:
                continue

            parse_results = haomo_annotation_parser(
                annotation_path, *bbox_filters)
            if parse_results is None:
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

        #  保存coco标注文件
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

    def merged_coco_format(self, card_id_root_list, merged_coco_format_save_root, json_name):
        all_train_image_format = list()
        all_train_annotation_format = list()
        train_image_counter = 0
        train_instance_counter = 0

        format_info = None
        format_license = None
        format_categories = None
        for i, card_id in enumerate(card_id_root_list):
            train_json_path = os.path.join(
                merged_coco_format_save_root, card_id, json_name)
            assert os.path.exists(train_json_path)
            with open(train_json_path, 'r') as fin:
                train_json = json.load(fin)

            if i == 0:
                format_info = train_json['info']
                format_license = train_json['license']
                format_categories = train_json['categories']
            else:
                assert train_json['info'] == format_info and train_json['license'] == format_license and train_json[
                    'categories'] == format_categories

            temp_old_image_id_to_new_image_id = {}
            train_images = train_json['images']
            for image_info in train_images:
                old_image_id = image_info['id']
                temp_old_image_id_to_new_image_id[old_image_id] = train_image_counter
                image_info['id'] = train_image_counter

                all_train_image_format.append(image_info)
                train_image_counter += 1

            train_annotations = train_json['annotations']
            for annotation in train_annotations:
                old_image_id = annotation['image_id']
                annotation['image_id'] = temp_old_image_id_to_new_image_id[old_image_id]
                annotation['id'] = train_instance_counter

                all_train_annotation_format.append(annotation)
                train_instance_counter += 1

        # 保存json
        print('Merge coco format ', json_name)
        train_coco_annotation_format = dict()
        train_coco_annotation_format['info'] = format_info
        train_coco_annotation_format['license'] = format_license
        train_coco_annotation_format['images'] = all_train_image_format
        train_coco_annotation_format['annotations'] = all_train_annotation_format
        train_coco_annotation_format['categories'] = format_categories
        with open(os.path.join(merged_coco_format_save_root, json_name), 'w') as fout:
            json.dump(train_coco_annotation_format, fout, indent=4)