import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import copy
from mmdet.datasets.api_wrappers import COCO, COCOeval


class CustomEval(COCOeval):
    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        在目标检测中 _.gts 索引Ann的index为 【图片ip， 类别ip】，得到的是一个list数组，如果一张图片的一个类别有多个bbox，
        那么list中将会有多个item ._dts同理
        :return: None
        '''

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        p = self.params
        if p.useCats:
            # 获取特定图片，特定类别的注释，主要是清除检测中出现gt中没有的img id，class id
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(
                imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(
                imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            # 部分比较小的物体，会设置忽略检测 根据json中的注释来定
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        # 给对应img，类别 添加对应的bbox信息
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        # 得到的是每张图片，单个类别的检测结果的集合。
        # per-image per-category evaluation results
        self.evalImgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print(
                'useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        # 取出GT中的，img cat id
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()     # 获取img为单位的检测结果与gt
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        # ious返回的是一个【M * N】的ndarry， 其中M是在这个img中，catId下有多少个预测的bbox， N是在这个img，catId下有多少个GT
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds
                     for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        # self.evalImages 顺序是 K，A，M，I 一共K*A*M*I个单张图片的检测结果，单张图片的特定类别，特定面积范围，特定最大检测个数下的检测结果。
        # 我们可以按照这个来索引对应的检测结果，在后续accumulate函数中有具体使用。
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    # 这块用cython写的，主要返回的就是 imgId，catId对应的M*N矩阵，每个值都是对应框的IoU值
    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            # 把这张图片的所有类别的所有检测结果进行一个数组的合并
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        # 按照网络预测的置信度score排序
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        # 把超出最大检测结果的bbox剔除
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + \
                        np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + \
                        np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / \
                    (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        计算本张图片，特定类别，特定面积阈值，特定最大检测结果下的result。
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            # 本张图片特定类别的所有检测结果与GT
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            # 如果不符合特定面积的阈值，就忽略
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        # gtind 前面都是 ignore为0 的gt 后面都是 ignore为1的gt
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        # 挑出满足我们这个特定area阈值下的所有gt
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        # 按照置信度大小挑出满足这个最大检测个数下的所有dt
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious

        # 得到满足area阈值的gt与所有dt的iou结果 （M * n（gtind））
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        # 得到我们需要设置的IoU阈值，超过定义为正样本，不符合则为负样本
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        # 在每个阈值下的Gt是否得到匹配
        gtm = np.zeros((T, G))
        # 在每个阈值下的Dt是否得到匹配
        dtm = np.zeros((T, D))
        # 所有忽略的gt
        gtIg = np.array([g['_ignore'] for g in gt])
        # 所有忽略的dt
        dtIg = np.zeros((T, D))

        # 如果这张图片存在这个类别的gt与dt
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):  # IoU index， IoU阈值
                # 按照置信度大小排序好的前 max_Det个dt
                for dind, d in enumerate(dt):
                    # 如果m= -1 代表这个dt没有得到匹配 m代表dt匹配的最好的gt的下标
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # 如果这个gt已经被其他置信度更好的dt匹配到了，本轮的dt就不能匹配这个gt了。
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # 因为gt已经按照ignore排好序了，前面的为0，于是当我们碰到第一个gt的ignore为1时，判断这个dt是否已经匹配到了
                        # 其他的gt，如果m>-1证明并且m对应的gt没有被ignore，就直接结束即可，对应的就是这个dt最好的gt。
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # 如果计算dt与gt的iou小于目前最佳的IoU，忽略这个gt
                        if ious[dind, gind] < iou:
                            continue
                        # 超过当前最佳的IoU，更新IoU与m的值
                        iou = ious[dind, gind]
                        m = gind
                    # 如果这个dt没有对应的gt与其匹配，继续dt的下一个循环
                    if m == -1:
                        continue
                    # 把当前dt与第m个gt进行匹配，修改dtm与gtm的值，分别一一对应
                    # 如果这个dt对应的最佳gt本身就是被ignore的，就把这个dt也设置为ignore。
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                     for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(
            dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)  # 多少个ioU的阈值
        R = len(p.recThrs)  # 多少个recall的阈值
        K = len(p.catIds) if p.useCats else 1  # 多少个类
        A = len(p.areaRng)  # 多少个面积阈值
        M = len(p.maxDets)  # 多少个最大检测数
        # -1 for the precision of absent categories
        precision = -np.ones((T, R, K, A, M))
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        # 对应不重复的K的id list 后续同此
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(
            map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)  # 多少个图片
        A0 = len(_pe.areaRng)  # 多少个面积阈值
        # retrieve E at each category, area range, and max number of detections
        # self.evalImgs 索引顺序是 K,A,M,I 所以找到在特定K，A，M下的所有图片，需要按照如下的三维索引
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0  # 当前K0前面过了多少图片与面积阈值
            for a, a0 in enumerate(a_list):
                Na = a0 * I0  # 在当前K0前面过了多少阈值
                for m, maxDet in enumerate(m_list):
                    # k0，a0下的所有Images
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    # k0，a0，maxdet下的所有Images的得分
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    # k0，a0，maxdet下所有Images得分从高到底的索引 inds
                    inds = np.argsort(-dtScores, kind='mergesort')
                    # 按照得分从高到低排序
                    dtScoresSorted = dtScores[inds]
                    # 在当前k0,a0下，每张图片不超过MaxDet的所有det按照ind排序。 dtm[T,sum(Det) in every imges]
                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet]
                                         for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    # 有多少个正样本
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    # 如果dtm对应的匹配gt不为0，且对应的gt没有被忽略，这个dt就是TP tips:[1,0,1,0,1,0]
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    # dtm对应的gt为0， 并且这个dt也没有被忽略，这个dt就是FP  tips:[0,1,0,1,0,1]
                    fps = np.logical_and(
                        np.logical_not(dtm), np.logical_not(dtIg))

                    # 按照行的方式（每个Iou阈值下）进行匹配到的累加 每个index也就是到这个置信度的时候有多少个tp，有多少个fp
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)  # 得到这个Iou下对应的tp tips:[1,0,2,0,3,0]
                        fp = np.array(fp)  # 得到这个IoU下对应的fp tips:[0,1,0,2,0,3]
                        nd = len(tp)  # 有多少个tp
                        # 每个置信度分数下对应的recall 如上述例子 若有3个正样本 则rc=[1/3,1/3,2/3,2/3,1,1]
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))  # 每个阶段对应的精度
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        # 当前i下的最大精度
                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        # 找到每个recall发生变化的时候的index，与p.recThrs一一对应，最接近其的值的index
                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                # 得到每个recall阈值对应的最大精度，存入q中
                                q[ri] = pr[pi]
                                # 得到这个recall值下的得分
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(
                            q)  # 按照recall的大小存入对应的精度
                        scores[t, :, k, a, m] = np.array(ss)  # 存入对应的分数
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            # 如果是'all' 就是所有尺度， 如果不是就是特定的尺度
            aind = [i for i, aRng in enumerate(
                p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            # 如果是ap，就从precision中得到对应面积阈值、最大检测数下的精度
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # 得到特定IoU下的所有pr
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]

            # 如果是recall，就取出recall的值
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            # 除去-1 其他的计算平均精度
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            # all iouThr， 所有recall下，所有面积下， 所有类别，在最大检测数100下的的平均AP
            stats[0] = _summarize(1)
            # [1]:IoU阈值为0.5 所有recall下，所有面积下， 所有类别，在最大检测数100下的的平均AP
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # [2]:IoU阈值为0.75 所有recall下，所有面积下， 所有类别，在最大检测数100下的的平均AP
            stats[2] = _summarize(
                1, iouThr=.75, maxDets=self.params.maxDets[2])
            # [3]: all iouThr， 所有recall下，small面积下， 所有类别，在最大检测数100下的的平均AP
            stats[3] = _summarize(1, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            # [4]: all iouThr， 所有recall下，medium面积下， 所有类别，在最大检测数100下的的平均AP
            stats[4] = _summarize(1, areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            # [5]: all iouThr， 所有recall下，large面积下， 所有类别，在最大检测数100下的的平均AP
            stats[5] = _summarize(1, areaRng='large',
                                  maxDets=self.params.maxDets[2])
            # [6]: all iouThr，所有面积下， 所有类别，在最大检测数1下的的平均recall
            stats[6] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[0])
            # [7]: all iouThr，所有面积下， 所有类别，在最大检测数10下的的平均recall
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # [8]: all iouThr，所有面积下， 所有类别，在最大检测数100下的的平均recall
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # [9]: all iouThr，small面积下， 所有类别，在最大检测数100下的的平均recall
            stats[9] = _summarize(0, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            # [10]: all iouThr，medium面积下， 所有类别，在最大检测数100下的的平均recall
            stats[10] = _summarize(0, areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            # [11]: all iouThr，large面积下， 所有类别，在最大检测数100下的的平均recall
            stats[11] = _summarize(
                0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()
