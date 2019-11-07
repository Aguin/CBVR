# 2019ByteCamp Group G1: 基于内容相似的视频检索算法

## 任务描述

比赛数据集为[FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K)，其中包含225960个视频，视频内容为Wikipedia中选取的4687个突发事故，另给定100个询问视频。选手需利用主办方预先抽取的视频关键帧特征，对在数据集上定义的Duplicate Scene Videos，Complementary Scene Videos，Incident Scene Videos三类视频进行检索，最终评价参考检索平均准确率mAP以及检索时间。

## 算法实现

分别基于哈希、量化、图索引、近似近邻等算法，尝试了直接检索以及粗排-精排的二级检索方法：

- Baseline：主办方给出的baseline以关键帧平均值暴力查找最近邻
- Accumulated Minimal Distance（AMD）：以视频之间[累积最小距离](http://yongyuan.name/blog/multi-frames-ranking-problem.html)查找最近邻
- HNSW：建立图索引进行一级检索，用AMD进行二级检索
- IVFPQ：倒排乘积量化进行以及检索，用AMD进行二级检索
- K-means：用K-means聚类中同一类作为一级检索结果，用AMD进行二级检索
- KNN：用KD-Tree和[Annoy](https://github.com/spotify/annoy)建立索引进行近似近邻检索，用AMD进行二级检索
- LSHash：以LSHash作为一级检索，分别尝试以AMD、Top-K阈值、最长公共子序列等作为二级检索
- MyLOPQ：以局部优化乘积量化作为一级检索，用AMD进行二级检索

## 实验结果

| **Method**                         | **DSVR_mAP** | **CSVR_mAP** | **ISVR_mAP** | **Time(sec.  Per query)** | **Construction**  **Time** |
| ---------------------------------- | ------------ | ------------ | ------------ | ------------------------- | -------------------------- |
| Baseline                           | 0.445        | 0.425        | 0.355        | 0.74s                     | -                          |
| DML                                | 0.425        | 0.405        | 0.332        | -                         | -                          |
| LBoW                               | **0.710**    | **0.675**    | **0.572**    | -                         | -                          |
| Accumulated  Minimal Distance(AMD) | 0.671        | 0.628        | 0.525        | 80~124s                   | -                          |
| KD-Tree(d=8)+AMD                   | 0.560        | -            | -            | 2~3s                      | 40 s                       |
| K-Means+AMD                        | 0.644        | 0.604        | 0.503        | >20s                      | 1  hr                      |
| IVFPQ+AMD                          | 0.634        | 0.592        | 0.492        | 4~15s                     | 52  s                      |
| HNSW+AMD                           | 0.640        | 0.597        | 0.495        | 2~3s                      | 30  min                    |
| LSH+AMD                            | 0.652        | 0.609        | 0.502        | 1.1s                      | 3.4  s                     |
| LSH+LCS(thres=0.5)                 | 0.590        | 0.550        | 0.450        | 2.2s                      | 3.4  s                     |
| LSH+Clip(thres=0.55)               | 0.639        | 0.596        | 0.496        | 1.5s                      | 3.4  s                     |
| LSH+Top15                          | 0.659        | 0.615        | 0.506        | 1.1s                      | 3.4  s                     |
| LSH+Top10                          | **0.663**    | **0.619**    | **0.507**    | 1.1s                      | **3.4  s**                 |
| LSH                                | 0.569        | 0.538        | 0.448        | **0.11s**                 | 3.4  s                     |