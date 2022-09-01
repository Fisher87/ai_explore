## 代码主流程:

```c++
cli_main.cc:
main()
     -> CLIRunTask()
          -> CLITrain()
               -> DMatrix::Load()
               -> learner = Learner::Create()
               -> learner->Configure()
               -> learner->InitModel()
               -> for (i = 0; i < param.num_round; ++i)
                    -> learner->UpdateOneIter()
                    -> learner->Save()
learner.cc:
Create()
      -> new LearnerImpl()
Configure()
InitModel()
     -> LazyInitModel()
          -> obj_ = ObjFunction::Create()
               -> objective.cc
                    Create()
                         -> SoftmaxMultiClassObj(multiclass_obj.cc)/
                              LambdaRankObj(rank_obj.cc)/
                              RegLossObj(regression_obj.cc)/
                              PoissonRegression(regression_obj.cc)
          -> gbm_ = GradientBooster::Create()
               -> gbm.cc
                    Create()
                         -> GBTree(gbtree.cc)/
                              GBLinear(gblinear.cc)
          -> obj_->Configure()
          -> gbm_->Configure()
UpdateOneIter()
      -> PredictRaw()
      -> obj_->GetGradient()
      -> gbm_->DoBoost()

gbtree.cc:
Configure()
      -> for (up in updaters)
           -> up->Init()
DoBoost()
      -> BoostNewTrees()
           -> new_tree = new RegTree()
           -> for (up in updaters)
                -> up->Update(new_tree)

tree_updater.cc:
Create()
     -> ColMaker/DistColMaker(updater_colmaker.cc)/
        SketchMaker(updater_skmaker.cc)/
        TreeRefresher(updater_refresh.cc)/
        TreePruner(updater_prune.cc)/
        HistMaker/CQHistMaker/
                  GlobalProposalHistMaker/
                  QuantileHistMaker(updater_histmaker.cc)/
        TreeSyncher(updater_sync.cc)
```

## 构建树流程:

`updater_colmaker.cc`

```c++
class Builder {
  public:
   // constructor
   explicit Builder(const TrainParam& param,
                    const ColMakerTrainParam& colmaker_train_param,
                    std::unique_ptr<SplitEvaluator> spliteval,
                    FeatureInteractionConstraintHost _interaction_constraints,
                    const std::vector<float> &column_densities)
       : param_(param), colmaker_train_param_{colmaker_train_param},
         nthread_(omp_get_max_threads()),
         spliteval_(std::move(spliteval)),
         interaction_constraints_{std::move(_interaction_constraints)},
         column_densities_(column_densities) {}
   // update one tree, growing
   virtual void Update(const std::vector<GradientPair>& gpair,
                       DMatrix* p_fmat,
                       RegTree* p_tree) {
     std::vector<int> newnodes;

     // 临时数据初始化
     this->InitData(gpair, *p_fmat, *p_tree);
     /*
        // 进行伯努利采样样本
        std::bernoulli_distribution coin_flip(param_.subsample);
        // 列采样, 即特征采样
        {
        column_sampler_.Init(fmat.Info().num_col_, param_.colsample_bynode,
                             param_.colsample_bylevel, param_.colsample_bytree);
        }
     */

     // initialize the base_weight, root_gain and NodeEntry for all the new nodes in qexpand ;
     // 为该level下可用于split的树节点计算统计量: gain/weight值, 初始情况下就是根节点;
     this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
     /*
        InitNewNode:
              // 计算每个树节点(已构建树)的一阶导数和(G) 以及二阶导数和(H);
              // G = \sum_(gi), H = \sum_(hi)
              for (int nid : qexpand) {
                GradStats stats;
                for (auto& s : stemp_) {
                  stats.Add(s[nid].stats);
                }
                // update node statistics
                snode_[nid].stats = stats;
              }

              // weight = - \sum_(gi) / (\sum_(hi) + lambda)
              -> ComputeWeight()
              // obj_fun(loss): [G*weight + (1/2)(H+lambda)*weight^2] + gamma * |weight|
              -> ComputeScore()
     */

     for (int depth = 0; depth < param_.max_depth; ++depth) {
       // 查找分割点
       this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
       /*
        FindSplit:
               -> for( each feature )  // 通过openMP 进行并行处理
                   -> UpdateSolution()
                       -> EnumerateSplit()    // 每个线程执行一个特征选出对应特征最优的分割值;
                                              // 在每个线程里汇总各个线程内分配到的数据样本对应的统计量: G(grad) / H(hess)
                                              // 然后每个线程计算出对应特征下最优分割点;
                   -> SyncBestSolution()  // 上面的UpdateSolution() 会为所有待扩展分割的叶结点找到特征
                                          // 维度的最优分割点，比如对于叶结点A，OpenMP线程1会找到特征f_1
                                          // 的最优分割点，OpenMP线程2会找到特征f_2的最优分割点, 所以需要
                                          // 进行全局sync，找到叶结点A的最优分割点。
                   
                   // 为需要进行分割的叶结点创建孩子结点, 并计算相应的孩子节点weight 值
                   for (int nid : qexpand) {
                       NodeEntry const &e = snode_[nid];
                       // now we know the solution in snode[nid], set split
                       if (e.best.loss_chg > kRtEps) {
                         bst_float left_leaf_weight =
                             spliteval_->ComputeWeight(nid, e.best.left_sum) *
                             param_.learning_rate;
                         bst_float right_leaf_weight =
                             spliteval_->ComputeWeight(nid, e.best.right_sum) *
                             param_.learning_rate;
                         p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                                            e.best.DefaultLeft(), e.weight, left_leaf_weight,
                                            right_leaf_weight, e.best.loss_chg,
                                            e.stats.sum_hess, 0);
                       } else {
                         (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate);
                       }
                     }
       */

       // 根据分割结果更新数据样本到树节点的映射关系, 处理缺失值样本;
       this->ResetPosition(qexpand_, p_fmat, *p_tree);

       // 将待扩展分割的叶子节点替换qexpand_, 准备下一轮的split处理;
       this->UpdateQueueExpand(*p_tree, qexpand_, &newnodes);

       // 为该level下可用于split的树节点计算统计量: gain/weight值 
       this->InitNewNode(newnodes, gpair, *p_fmat, *p_tree);

       for (auto nid : qexpand_) {
         if ((*p_tree)[nid].IsLeaf()) {
           continue;
         }
         int cleft = (*p_tree)[nid].LeftChild();
         int cright = (*p_tree)[nid].RightChild();
         spliteval_->AddSplit(nid,
                              cleft,
                              cright,
                              snode_[nid].best.SplitIndex(),
                              snode_[cleft].weight,
                              snode_[cright].weight);
         interaction_constraints_.Split(nid, snode_[nid].best.SplitIndex(), cleft, cright);
       }
       qexpand_ = newnodes;
       // if nothing left to be expand, break
       if (qexpand_.size() == 0) break;
     }
     // set all the rest expanding nodes to leaf
     for (const int nid : qexpand_) {
       (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
     }
     // remember auxiliary statistics in the tree node
     for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
       p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
       p_tree->Stat(nid).base_weight = snode_[nid].weight;
       p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
     }
   }
```

#### 参考博客:
1.[xgboost源码之Registry](https://datavalley.github.io/2017/09/18/xgboost%E6%BA%90%E7%A0%81%E4%B9%8BRegistry)
