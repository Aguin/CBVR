from sklearn import datasets# 导入内置数据集模块
import numpy as np
from sklearn.neighbors import KDTree#导入KD树类

## Use KNN to Return a Small List for Each Query
def knn_init(features):
    np.random.seed(19260817)  # 设置随机种子，不设置的话默认是按系统时间作为参数，因此每次调用随机模块时产生的随机数都不一样设置后每次产生的一样
    tree = KDTree(np.array(features))
    return tree

def knn_inference(tree,query_features, final_vids, features,id_index):
    """
      Use Knn to Return a Small List for Each Query
      Args:
        tree : kdtreeClass
        query_features: N*D
        final_vids: M*1
        features: M*D
        id_index: dict, (vid, [st_index, ed_index])
      Returns:
        small_vids: Q*1
        small_features: Q*D
    """
    query_features=query_features[:,0:16]
    set_vids = set()
    #print(query_features)
    for query_feature in query_features:
        #print(query_feature.reshape(1,-1).shape)
        dist, ind = tree.query(query_feature.reshape(1,-1), k=800)
        ind=ind.tolist()[0]
        for x in ind:
            set_vids.add(final_vids[x])
    #print(set_vids)
    #print(final_vids)
    #print(features)
    small_vids = []
    small_features = []
    for vid in set_vids:
        small_vids.extend(final_vids[id_index[vid][0]:id_index[vid][1]])
        small_features.extend(features[id_index[vid][0]:id_index[vid][1]])

    return small_vids,small_features
