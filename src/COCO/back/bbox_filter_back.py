import math


def bbox_filter_square(bbox):
    """
    :param bbox: dict in annotation json
    :return:
    """
    if 'bbox' in bbox:
        if (max(bbox['bbox'][2], bbox['bbox'][3]) / min(bbox['bbox'][2], bbox['bbox'][3])) < 1.4:
            #长边比短边小于1.4的滤掉
            return True
        else:
            return False
    else:
        return True  # bbox标注都没有，无法训练检测


def bbox_filter_sublights_0(bbox):
    if bbox['characteristic'] == 0 and bbox['num_sub_lights'] == 0 and bbox['toward_orientation'] != 0:
        # 通行灯，且子灯数量为0且不正对着的灯滤掉
        return True
    else:
        return False


def bbox_filter_size_4(bbox):
    """
    :param bbox: dict in annotation json
    :return:
    """
    if 'bbox' in bbox:
        if math.ceil(bbox['bbox'][2]) < 4 or math.ceil(bbox['bbox'][3]) < 4:
            # 任意一边小于4滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_size_8(bbox):
    """
    :param bbox: dict in annotation json
    :return:
    """
    if 'bbox' in bbox:
        if math.ceil(bbox['bbox'][2]) < 8 or math.ceil(bbox['bbox'][3]) < 8:
            # 任意一边小于8 滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_size_10(bbox):
    """
    :param bbox: dict in annotation json
    :return:
    """
    if 'bbox' in bbox:
        if math.ceil(bbox['bbox'][2]) < 10 or math.ceil(bbox['bbox'][3]) < 10:
            # 任意一边小于10滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_size_14(bbox):
    """
    :param bbox: dict in annotation json
    :return:
    """
    if 'bbox' in bbox:
        if math.ceil(bbox['bbox'][2]) < 14 or math.ceil(bbox['bbox'][3]) < 14:
            # 任意一边小于14滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_size_16(bbox):
    """
    :param bbox: dict in annotation json
    :return:
    """
    if 'bbox' in bbox:
        if math.ceil(bbox['bbox'][2]) < 16 or math.ceil(bbox['bbox'][3]) < 16:
            # 任意一边小于16滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_ratio(bbox):
    if 'bbox' in bbox:
        if (max(bbox["bbox"][2], bbox["bbox"][3]) / min(bbox["bbox"][2], bbox["bbox"][3])) < 1.6:
            #长边/短边小于1.6滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_truncation(bbox):
    """
    通过object中的key 'truncation' 截断的红绿灯
    :param bbox: dict in annotation json
    :return:
    """
    if 'truncation' in bbox:
        if bbox['truncation'] == 1:
            #截断的灯滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_ext_occlusion_sub_number(bbox):
    """
    通过object中的key 'ext_occlusion' 过滤外遮挡的红绿灯
    :param bbox: dict in annotation json
    :return:
    """
    if 'ext_occlusion' in bbox:
        if bbox['ext_occlusion'] == 1 and bbox['num_sub_lights'] < 3:
            #被外物遮挡且子灯数量小于3滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_ext_occlusion(bbox):
    """
    通过object中的key 'ext_occlusion' 过滤外遮挡的红绿灯
    :param bbox: dict in annotation json
    :return:
    """
    if 'ext_occlusion' in bbox:
        if bbox['ext_occlusion'] == 1:
            #被外物遮挡过滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_pose_orientation(bbox):
    if 'pose_orientation' in bbox:
        if bbox['pose_orientation'] not in [0, 1]:
            # 非横向和竖向的灯滤掉
            return True
        else:
            return False
    else:
        return False


def bbox_filter_toward_orientation(bbox):
    if 'toward_orientation' in bbox:
        if bbox['toward_orientation'] not in [0, 1]:  # V18背面标的是1, V21背面标的是2
            return True
        else:
            return False
    else:
        return True


def bbox_filter_toward_orientation_front(bbox):
    if 'toward_orientation' in bbox:
        if bbox['toward_orientation'] not in [0]:  # V18背面标的是1, V21背面标的是2
            #非正对的灯滤掉
            return True
        else:
            return False
    else:
        return True


def bbox_filter_characteristic(bbox):
    # V18 没有 characteristic 字段所以应该 return false
    if 'characteristic' in bbox:
        if bbox['characteristic'] not in [0, 1]:
            #非通行灯和行人灯滤掉
            return True
        else:
            return False
    else:
        return False


def bbox_filter_characteristic_cars(bbox):
    # V18 没有 characteristic 字段所以应该 return false
    # 0 - 通行灯 | 1 - 行人灯 | 2 - 数字灯 | 3 - 符号 & 文字灯 | 4 - 其它灯 | 5 - 未知
    if 'characteristic' in bbox:
        if bbox['characteristic'] not in [0]:
            #非通行灯滤掉
            return True
        else:
            return False
    else:
        return False


def bbox_filter_without_yellow_light(bbox):
    if 'sub_lights' in bbox:
        for sub_light in bbox['sub_lights']:
            if sub_light['color'] == 1:
                return False
    return True
