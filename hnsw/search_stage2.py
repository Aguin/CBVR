import numpy as np
import pickle as pk
from collections import defaultdict
from scipy.spatial.distance import cdist


def calc_ave_min_dist(query_name, query_features, selected_vids, selected_features, vid2names):
    # list转为numpy数组
    query_features = np.array(query_features).reshape(-1, 512)
    selected_features = np.array(selected_features).reshape(-1, 512)
    # 计算询问关键帧和所有粗排选出关键帧的距离
    dist = np.nan_to_num(cdist(query_features, selected_features, metric='cosine'))
    # 建立每个vid对应的索引列表映射
    vid2cols = defaultdict(list)
    for i in range(len(selected_features)):
        vid2cols[selected_vids[i]].append(i)
    # 返回query_result = {"name": sim}, del query_name
    query_result = {}
    for vid in vid2cols.keys():
        cols = vid2cols[vid]
        vid_dist = dist[:, cols]
        # 计算累计最小距离
        dmin = np.average(np.min(vid_dist, axis=1))
        # 计算相似度
        sim = 1 - dmin
        name = vid2names[vid][0]
        query_result[name] = sim
    if query_name in query_result:
        del query_result[query_name]
    return query_result

def calc_topk_min_dist(query_name, query_features, selected_vids, selected_features, vid2names, k=5):
    # list转为numpy数组
    query_features = np.array(query_features).reshape(-1, 512)
    selected_features = np.array(selected_features).reshape(-1, 512)
    # 计算询问关键帧和所有粗排选出关键帧的距离
    dist = np.nan_to_num(cdist(query_features, selected_features, metric='cosine'))
    # 建立每个vid对应的索引列表映射
    vid2cols = defaultdict(list)
    for i in range(len(selected_features)):
        vid2cols[selected_vids[i]].append(i)
    # 返回query_result = {"name": sim}, del query_name
    query_result = {}
    for vid in vid2cols.keys():
        cols = vid2cols[vid]
        vid_dist = dist[:, cols]
        # 计算累计最小距离
        dmin = np.average(np.sort(np.min(vid_dist, axis=1))[:k])
        # 计算相似度
        sim = 1 - dmin
        name = vid2names[vid][0]
        query_result[name] = sim
    if query_name in query_result:
        del query_result[query_name]
    return query_result

def calc_mean_dist(query_name, query_features, selected_vids, selected_features, vid2names):
    # list转为numpy数组
    query_features = np.array(query_features).reshape(-1, 512)
    selected_features = np.array(selected_features).reshape(-1, 512)
    # 计算询问关键帧和所有粗排选出关键帧的距离
    dist = np.nan_to_num(cdist(query_features, selected_features, metric='cosine'))
    # 建立每个vid对应的索引列表映射
    vid2cols = defaultdict(list)
    for i in range(len(selected_features)):
        vid2cols[selected_vids[i]].append(i)
    # 返回query_result = {"name": sim}, del query_name
    query_result = {}
    for vid in vid2cols.keys():
        cols = vid2cols[vid]
        vid_dist = dist[:, cols]
        # 计算累计最小距离
        dmin = np.average(vid_dist)
        # 计算相似度
        sim = 1 - dmin
        name = vid2names[vid][0]
        query_result[name] = sim
    if query_name in query_result:
        del query_result[query_name]
    return query_result

def calc_lcs_dist(query_name, query_features, selected_vids, selected_features, vid2names, threshold=0.50):
    # list转为numpy数组
    query_features = np.array(query_features).reshape(-1, 512)
    selected_features = np.array(selected_features).reshape(-1, 512)
    # 计算询问关键帧和所有粗排选出关键帧的距离
    dist = np.nan_to_num(cdist(query_features, selected_features, metric='cosine'))
    # 建立每个vid对应的索引列表映射
    vid2cols = defaultdict(list)
    for i in range(len(selected_features)):
        vid2cols[selected_vids[i]].append(i)
    # 返回query_result = {"name": sim}, del query_name
    query_result = {}
    for vid in vid2cols.keys():
        cols = vid2cols[vid]
        vid_dist = dist[:, cols]
        # 计算累计最小距离
        n,m=vid_dist.shape
        dp=[[0 for j in range(m)] for i in range(n)]
        for i in range(n):
            for j in range(m):
                if i>0:
                    dp[i][j]=max(dp[i][j],dp[i-1][j])
                if j>0:
                    dp[i][j]=max(dp[i][j],dp[i][j-1])
                if 1-vid_dist[i][j]>=threshold:
                    dp[i][j]=max(dp[i][j],dp[i-1][j-1]+1-vid_dist[i][j])
        # dmin = np.average(np.sort(np.min(vid_dist, axis=1))[:k])
        # 计算相似度
        #sim = 1 - dmin
        name = vid2names[vid][0]
        query_result[name] = dp[n-1][m-1]/n
    if query_name in query_result:
        del query_result[query_name]
    return query_result

if __name__ == "__main__":
    with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
        vid2names = pk.load(pk_file)
    query_name = "F5olduWiQ24"
    query_features = [np.random.rand(512) for _ in range(10)]
    vids = [
        'v0202dba0000bjuhtafm1hf6r257u2o0',
        'v0202dba0000bjuhtvcttc5qicghbeb0',
        'v0202dba0000bjuhv9r82vu4nmcr2e8g',
        'v0202dba0000bjuhvhvvrnhtc3gt32eg',
        'v0202dba0000bjui08j82vu4nmcr2npg',
        'v0202dba0000bjui2bj82vu4nmcr3bi0',
        'v0202dba0000bjui34r82vu4nmcr3itg',
        'v0202dba0000bjui3fnvrnhtc3gt464g',
        'v0202dba0000bjui3n7vrnhtc3gt480g',
        'v0202dba0000bjui3o382vu4nmcr3ou0',
        'v0202dba0000bjui41cttc5qicghd5v0',
        'v0202dba0000bjui48vm1hf6r2580440',
        'v0202dba0000bjui4n7m1hf6r258087g',
        'v0202dba0000bjui578a2pekioqjkd30'
    ]
    selected_vids = []
    selected_features = []
    for vid in vids:
        l = np.random.randint(20, 30)
        selected_vids += [vid] * l
        selected_features += [np.random.rand(512) for _ in range(l)]
    query_result = calc_mean_dist(query_name, query_features, selected_vids, selected_features, vid2names)
    print(query_result)