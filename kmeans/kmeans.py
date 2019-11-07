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
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

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


def kmeans_inference(kmeans, query_features, lables, final_vids, features, id_index):
    """
      Use K-Means to Return a Small List for Each Query
      Efficient implementation
      Args:
        query_features: N*D
        lables: NumClass*NumInClass
        final_vids: M*1
        features: M*D
        id_index: dict, (vid, [st_index, ed_index])
      Returns:
        cluster_vids: Q*1
        cluster_features: Q*D
    """
    query_lables = kmeans.predict(query_features)
    set_vids = set(final_vids[sum(lables[query_lables],[])])

    cluster_vids = []
    cluster_features = []
    for vid in set_vids:
        cluster_vids.extend(final_vids[id_index[vid][0]:id_index[vid][1]])
        cluster_features.extend(features[id_index[vid][0]:id_index[vid][1]])

    return cluster_vids, cluster_features

# ## Use K-Means to Return a Small List for Each Query
# def kmeans_inference(kmeans, query_features, lables, final_vids, features):
#     set_vids = []
#     for query_feature in query_features:
#         query_lable = kmeans.predict(np.expand_dims(query_feature, axis=0))
#         matched_vids = final_vids[lables==query_lable]
#         for vid in matched_vids:
#             if vid not in set_vids:
#                 set_vids.append(vid)
#     cluster_vids = []
#     cluster_features = []
#     for vid in set_vids:
#         # tmp = np.where(final_vids==vid)
#         # cluster_vids.extend(final_vids[final_vids==vid])
#         # cluster_features.extend(features[final_vids==vid])
#         cluster_vids.extend(final_vids[final_vids==vid])
#         cluster_features.extend(features[final_vids==vid])

#     return cluster_vids, cluster_features

if __name__ == '__main__':
    gtobj = GTOBJ()
    relevant_labels_mapping = {
        'DSVR': ['ND','DS'],
        'CSVR': ['ND','DS','CS'],
        'ISVR': ['ND','DS','CS','IS'],
    }

    start = time.time()
    final_vids, features, vid2features = load_features('/home/camp/FIVR/features/vcms_v1', is_gv=False)
    print('Read time: %.2f'% (time.time() - start))

    ## Calculate MiniBatch K-Means
    features = np.asarray(features, np.float32)
    print(features.shape)
    final_vids = np.array(final_vids)
    start = time.time()
    kmeans = MiniBatchKMeans(n_clusters=1024, init='k-means++', batch_size=1000, n_init=10, 
                    max_iter=3000, verbose=1, random_state=0, reassignment_ratio=0.1).fit(features)
    print('Kmeans time: %.2f'% (time.time() - start))
    joblib.dump(kmeans, './kmeans/kmeans.pkl')
    lables = kmeans.predict(features)
    print(type(lables))
    np.save('./kmeans/kmeans_lables.npy',lables)

    # ## Load K-Means Model
    # kmeans = joblib.load('./kmeans/kmeans.pkl')
    # lables = np.load('./kmeans/kmeans_lables.npy')
    # with open('./kmeans/kmeans_lables.json', 'w') as f:
    #     json.dump(lables, f)
    # with open('./kmeans/kmeans_lables.json', 'r') as f:
    #     lables = json.load(f)

    # # 加载vid2name 和 name2vid
    # with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    #     vid2names = pk.load(pk_file)
    # with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    #     name2vids = pk.load(pk_file)

    # # 开始评估
    # annotation_dir = '/home/camp/FIVR/annotation'
    # final_names = np.asarray([vid2names[vid][0] for vid in final_vids])
    # query_names = None
    # results = None
    # for task_name in ['DSVR', 'CSVR', 'ISVR']:
    #     annotation_path = os.path.join(annotation_dir, task_name + '.json')
    #     with open(annotation_path, 'r') as annotation_file:
    #         json_obj = json.load(annotation_file)
    #     if results is not None:
    #         continue
    #     query_names = json_obj.keys()
    #     query_names = [str(query_name) for query_name in query_names]
    #     results = dict()
    #     for query_name in query_names:
    #         start = time.time()
    #         query_features = features[final_names == query_name]
    #         print(type(query_features), query_features.shape)
    #         ## Stage-1
    #         cluster_vids, cluster_features = kmeans_inference(kmeans, query_features, lables, final_vids, features)
    #         print(len(cluster_vids), len(cluster_features))
    #         print('Kmeans Inference time: %.2f'% (time.time() - start))
    #     #     ## Stage-2
    #     #     query_result = STAGE2()
    #     #     ## Add to result
    #     #     results[query_name] = query_result

    #     # mAPOffcial, precisions = evaluateOfficial(annotations=gtobj.annotations, results=results,
    #     #                                           relevant_labels=relevant_labels_mapping[task_name],
    #     #                                           dataset=gtobj.dataset,
    #     #                                           quiet=False)
    #     # print('{} mAPOffcial is {}'.format(task_name, np.mean(mAPOffcial)))








