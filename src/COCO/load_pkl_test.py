import pickle
import numpy as np
from torch import float32

a = np.array([1, 2, 3, 4, 5])
print (a)
b = np.zeros((2, 3))
print (b)
c = np.arange(10)
print (c)
d = np.arange(2, 10, dtype=np.float32)
print (d)
e = np.linspace(1.0, 4.0, 6)
print (e)
f = np.indices((3, 3))
print (f)

data = ["label1", "label2", "label3", "label4", "label_i"]
array_temp = np.array(data)
array_temp.flatten()
print(array_temp)
with open("/home/fengzhen/fengzhen_ssd/data_set_python_script/trafficlight_check_script/data/QA/HD_2M_0_1999/medicine.pkl", "wb") as f:
    # pickle.dump(data, f)
    pickle.dump(array_temp, f)

# open的参数是pkl文件的路径
# fr = open("/home/fengzhen/fengzhen_ssd/data_set_python_script/trafficlight_check_script/data/HD_2M_0_1999/results_fz.pkl", 'rb')
fr = open("/home/fengzhen/fengzhen_ssd/data_set_python_script/trafficlight_check_script/data/QA/HD_2M_0_1999/medicine.pkl", 'rb')
inf = pickle.load(fr)  # 读取pkl文件的内容
print(inf)
fr.close()

# 转换成pkl格式示例
def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)
        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
