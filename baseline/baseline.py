## -*- coding: utf-8 -*-
import numpy as np
import os
import h5py
from tqdm import tqdm
from glob import glob
import pickle as pk
import json
import time
from scipy.spatial.distance import cdist
from future.utils import viewitems, lrange
from sklearn.metrics import precision_recall_curve

def read_h5file(path):
    hf = h5py.File(path, 'r')
    g1 = hf.get('images')
    g2 = hf.get('names')
    return g1.keys(), g1, g2
def load_features(dataset_dir, is_gv=True):
    '''
    加载特征
    :param dataset_dir: 特征所在的目录, 例如：/home/camp/FIVR/features/vcms_v1
    :param is_gv: 是否取平均。True：返回帧平均的结果，False：保留所有帧的特征
    :return:
    '''
    h5_paths = glob(os.path.join(dataset_dir, '*.h5'))
    print(h5_paths)
    vid2features = {}
    final_vids = []
    features = []
    for h5_path in h5_paths:
        vids, g1, g2 = read_h5file(h5_path)
        for vid in tqdm(vids):
            if is_gv:
                cur_arr = g1.get(vid)
                cur_arr = np.mean(cur_arr, axis=0, keepdims=False)
                cur_arr /= (np.linalg.norm(cur_arr, ord=2, axis=0))
                vid2features[vid] = cur_arr
            else:
                cur_arr = g1.get(vid)
                cur_arr = np.concatenate([cur_arr, np.mean(cur_arr, axis=0, keepdims=True)], axis=0)
                vid2features[vid] = cur_arr
                final_vids.extend([vid] * len(cur_arr))
                features.extend(cur_arr)
    if is_gv:
        return vid2features
    else:
        return final_vids, features, vid2features
def calculate_similarities(query_features, all_features):
    """
      用于计算两组特征(已经做过l2-norm)之间的相似度
      Args:
        queries: shape: [N, D]
        features: shape: [M, D]
      Returns:
        similarities: shape: [N, M]
    """
    similarities = []
    # 计算待查询视频和所有视频的距离
    dist = np.nan_to_num(cdist(query_features, all_features, metric='cosine'))
    for i, v in enumerate(query_features):
        # 归一化，将距离转化成相似度
        # sim = np.round(1 - dist[i] / dist[i].max(), decimals=6)
        sim = 1-dist[i]
        # 按照相似度的从大到小排列，输出index
        similarities += [[(s, sim[s]) for s in sim.argsort()[::-1] if not np.isnan(sim[s])]]
    return similarities
def evaluateOfficial(annotations, results, relevant_labels, dataset, quiet):
    """
      Calculate of mAP and interpolated PR-curve based on the FIVR evaluation process.
      Args:
        annotations: the annotation labels for each query
        results: the similarities of each query with the videos in the dataset
        relevant_labels: labels that are considered positives
        dataset: video ids contained in the dataset
      Returns:
        mAP: the mean Average Precision
        ps_curve: the values of the PR-curve
    """
    pr, mAP = [], []
    iterations = viewitems(annotations) if not quiet else tqdm(viewitems(annotations))
    for query, gt_sets in iterations:
        query = str(query)
        if query not in results: print('WARNING: Query {} is missing from the result file'.format(query)); continue
        if query not in dataset: print('WARNING: Query {} is not in the dataset'.format(query)); continue

        # set of relevant videos
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt: print('WARNING: Empty annotation set for query {}'.format(query)); continue

        # calculation of mean Average Precision (Eq. 6)
        i, ri, s = 0.0, 0, 0.0
        y_target, y_score = [], []
        for video, sim in sorted(viewitems(results[query]), key=lambda x: x[1], reverse=True):
            if video in dataset:
                y_score.append(sim)
                y_target.append(1.0 if video in query_gt else 0.0)
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        mAP.append(s / len(query_gt))
        if not quiet:
            print('Query:{}\t\tAP={:.4f}'.format(query, s / len(query_gt)))

        # add the dataset videos that are missing from the result file
        missing = len(query_gt) - y_target.count(1)
        y_target += [1.0 for _ in lrange(missing)] # add 1. for the relevant videos
        y_target += [0.0 for _ in lrange(len(dataset) - len(y_target))] # add 0. for the irrelevant videos
        y_score += [0.0 for _ in lrange(len(dataset) - len(y_score))]

        # calculation of interpolate PR-curve (Eq. 5)
        precision, recall, thresholds = precision_recall_curve(y_target, y_score)
        p = []
        for i in lrange(20, -1, -1):
            idx = np.where((recall >= i * 0.05))[0]
            p.append(np.max(precision[idx]))
        pr.append(p)
    # return mAP
    return mAP, np.mean(pr, axis=0)[::-1]
class GTOBJ:
    def __init__(self):
        annotation_path = '/home/camp/FIVR/annotation/annotation.json'
        dataset_path = '/home/camp/FIVR/annotation/youtube_ids.txt'
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        self.dataset = set(np.loadtxt(dataset_path, dtype=str).tolist())
gtobj = GTOBJ()
relevant_labels_mapping = {
    'DSVR': ['ND','DS'],
    'CSVR': ['ND','DS','CS'],
    'ISVR': ['ND','DS','CS','IS'],
}

vid2features = load_features('/home/camp/FIVR/features/vcms_v1', is_gv=True)


# 加载特征
vids = list(vid2features.keys())
print(vids[:10])
global_features = np.squeeze(np.asarray(list(vid2features.values()), np.float32))
print(np.shape(global_features))


# 加载vid2name 和 name2vid
with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    vid2names = pk.load(pk_file)
with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    name2vids = pk.load(pk_file)

# 开始评估
annotation_dir = '/home/camp/FIVR/annotation'
names = np.asarray([vid2names[vid][0] for vid in vids])
query_names = None
results = None
for task_name in ['DSVR', 'CSVR', 'ISVR']:
    annotation_path = os.path.join(annotation_dir, task_name + '.json')
    with open(annotation_path, 'r') as annotation_file:
        json_obj = json.load(annotation_file)
    if results is None:
        query_names = json_obj.keys()
        query_names = [str(query_name) for query_name in query_names]
        query_indexs = []
        for query_name in query_names:
            tmp = np.where(names == query_name)
            if len(tmp) != 0 and len(tmp[0]) != 0:
                query_indexs.append(tmp[0][0])
            else:
                print('skip query: ', query_name)
        query_features = np.squeeze(global_features[query_indexs])
        start = time.time()
        similarities = calculate_similarities(query_features, global_features)
        results = dict()
        for query_idx, query_name in enumerate(query_names):
            cur_sim = similarities[query_idx]
            query_result = dict(
                map(lambda v: (names[v[0]], v[1]), cur_sim)
            )
            del query_result[query_name]
            results[query_name] = query_result
        print('Total time: %.4f' % ((time.time() - start)/100))
    mAPOffcial, precisions = evaluateOfficial(annotations=gtobj.annotations, results=results,
                                              relevant_labels=relevant_labels_mapping[task_name],
                                              dataset=gtobj.dataset,
                                              quiet=False)
    print('{} mAPOffcial is {}'.format(task_name, np.mean(mAPOffcial)))

