**AI_EXPLORE**

![](https://readthedocs.org/projects/pygorithm/badge/?version=latest) ![](https://img.shields.io/badge/python%20-%203.7-brightgreen.svg)
========
> 包含机器学习、深度学习基础知识. 推荐系统及nlp相关算法;

## [机器学习](https://github.com/Fisher87/ai_explore/tree/master/ml_explore)

#### [machine learning](https://github.com/Fisher87/ai_explore/tree/master/ml_explore/machineLearning)
##### [有监督模型](https://github.com/Fisher87/ai_explore/tree/master/ml_explore/machineLearning/supervised)
+ [decision_tree](https://github.com/Fisher87/ai_explore/blob/master/ml_explore/machineLearning/supervised/decision_tree.py)
+ [gbdt](https://github.com/Fisher87/ai_explore/blob/master/ml_explore/machineLearning/supervised/gbdt.py)
+ [lr](https://github.com/Fisher87/ai_explore/blob/master/ml_explore/machineLearning/supervised/ls.py)
+ [hmm/crf](https://github.com/Fisher87/ai_explore/blob/master/ml_explore/machineLearning/supervised/hmm.py)
###### 常用算法
+ [viterbi解码算法](https://github.com/Fisher87/ai_explore/blob/master/ml_explore/machineLearning/supervised/viterbi.py)

##### [无监督模型](https://github.com/Fisher87/ai_explore/tree/master/ml_explore/machineLearning/unsupervised)
+ EM算法

#### [deep learning](https://github.com/Fisher87/ai_explore/tree/master/ml_explore/deepLearning)

---

## [自然语言处理](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore)
#### [基础模型实现](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/basic_model)
+ [DNN](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/basic_model/dnn.py)
+ [CNN](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/basic_model/cnn.py)
+ [RNN/LSTM](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/basic_model/rnn.py)
+ [Transformer](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/basic_model/transformer.py)
+ [HighWay](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/basic_model/highway.py)
> 基础模型常用函数模块
+ [modules](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/basic_model/modules.py)

#### [常用nlp任务](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/task)
+ [文本分类(classify)](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/task/classify)
    + TextCNN
    + Bilstm_Att
+ [实体识别(ner)](http://github.com/Fisher87/ai_explore/tree/master/nlp_explore/task/ner)
    + HMM
    + CRF
    + Bilstm_CRF
+ [文本匹配(text_match)](http://github.com/Fisher87/ai_explore/tree/master/nlp_explore/task/text_match):
  + [文本相关算法](http://github.com/Fisher87/ai_explore/tree/master/nlp_explore/task/text_match/lexical):
    + BM25
    + LSI
    + TFIDF  
  + [语义相关算法](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/task/text_match/semantic):
    + DSSM
    + ABCNN
    + TPConvNet
    + ESIM
    + BIMPM
    + DIIN
    + DRCN
    + BERT_Match
  
 + [ChatBot](https://github.com/Fisher87/ai_explore/tree/master/nlp_explore/task/chatbot)
    + Seq2Seq_Att

---

## [推荐系统](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore)
#### [常用召回策略](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/recall)
+ [CF](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/recall/CF):
  + item_based
  + user_based
  + graph

+ [LMF](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/recall/LMF)
  + ALS
  + MF

+ [DeepMatch](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/recall/DeepMatch):

#### [常用排序策略](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/rank)
+ [基于传统机器学习](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/rank/classical_ml)
    + LR
    + [MLR](https://zhuanlan.zhihu.com/p/100532677)
    + GBDT+LR
    + FM/FFM

+ [基于深度算法](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/rank/deep_dl)
    + Wide_Deep
    + DeepFM
    + PNN
    + FNN
    + DCN
    + DIN
    + DIEN

+ [在线训练](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/rank/online)
    + FTRL

#### [竞赛及其解决方案](https://github.com/Fisher87/ai_explore/tree/master/recommend_explore/competition)

---

## [理论思维导图](https://github.com/Fisher87/ai_explore/tree/master/xmind)
+ [传统机器学习.xmind](https://github.com/Fisher87/ai_explore/blob/master/xmind/%E4%BC%A0%E7%BB%9F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0.xmind)
+ [深度学习.xmind](https://github.com/Fisher87/ai_explore/blob/master/xmind/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0.xmind)
+ [自然语言处理.xmind](https://github.com/Fisher87/ai_explore/blob/master/xmind/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86.xmind)
+ [推荐系统.xmind](https://github.com/Fisher87/ai_explore/blob/master/xmind/%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F.xmind)

