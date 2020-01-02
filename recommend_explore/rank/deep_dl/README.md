#### [1.FNN(Factorisation-machine supported Neural Networks)]()

> [paper](https://arxiv.org/pdf/1601.02376.pdf)

![FNN模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/FNN.png)

`FNN`假设输入数据的格式是离散的类别特征(表示为`one-hot`编码)，且**每个特征属于一个`field`**，通过`embedding层`将高纬稀疏特征映射成低维稠密特征后，再作为多层感知机(MLP)的输入。
一般来说，embedding 层的参数可以随机初始化，**但是在FNN中，初始化embedding是采用通过FM预训练得到的每个特征的隐向量**，这样初始化的好处是将预训练的向量作为初始化参数时，能够让模型的参数在初始化的时候就处于较优的位置(训练的目的其实就是为了得到最优的模型参数)，能够加快收敛的过程，至于效果方面，则不一定会优于随机初始化的情况，因为随机初始化经过多轮的迭代也可能会收敛同样的效果。

------

#### [2.PNN]()

> [paper](https://arxiv.org/pdf/1611.00144.pdf)

![PNN模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/PNN.png)

`PNN`是在`FNN`的基础上进行改进，就是**增加了特征的二阶交叉项**. 因此, `FNN`和`PNN`的关系，类似于`LR`和`FM`的关系，只是`FNN`和`PNN`均是对原始特征进行了`embedding`映射, 其实PNN里面的主要贡献就是在Embedding的基础上再多做一些**Pairwise Product**(包含Inner Product和Outer Product)的操作增强高阶/非线性效果;

------

#### [3.DeepCrossing]()
> [paper](https://arxiv.org/pdf/1708.05123.pdf)  

![DeepCross模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/DeepCross.png)

PNN 进行了特征的二阶交叉，目前是为了获得信息量更多的特征，除了二阶，三阶四阶甚至更高阶的特征会更加有区分度；Deep&Cross 就是一个能够进行任意高阶交叉的神经网络;

------


#### [4.WideDeep]()
> [paper](https://arxiv.org/pdf/1606.07792.pdf) 

![WideDeep模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/WideDeep.png)

`Wide`部分其实就是LR, `Deep`部分就是`FNN`, 只是deep部分中的embedding层不用FM训练得到的隐向量初始化。wide部分主要负责memorization, deep部分主要负责generalization. memorization 主要指的是记住出现过的样本，可以理解为拟合训练数据的能力，generalization 则是泛化能力;

*根据论文的实验，wide & deep 比起单纯的 wide 或 deep 都要好，但是根据我后面的实验以及网上的一些文章，wide 部分仍然需要人工设计特征，在特征设计不够好的情况下，wide&deep 整个模型的效果并不如单个的 deep 模型。*

Wide&Deep 中还允许输入连续的特征，这点与 FNN 不同，连续特征可以直接作为 Wide 部分或 Deep 部分的输入而无需 embedding 的映射，

------

#### [5.DeepFM]()
> [paper](https://arxiv.org/pdf/1703.04247.pdf)

![DeepFM模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/DeepFM.png)

`DeepFM`其实就是模仿`Wide&Deep`, 只是将`Wide`部分替换成了`FM`;

------

#### [6.DIN]()
> [paper](https://arxiv.org/pdf/1706.06978.pdf)

![DIN模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/DIN.png)

从之前提到的几个模型可知，CTR预估中的深度学习模型的基本思路是将原始的高维稀疏特征映射到一个低维空间中，也即对原始特征做了embedding操作，之后一起通过一个全连接网络学习到特征间的交互信息和最终与CTR之间的非线性关系。这里值得注意的一点是，在对用户历史行为数据进行处理时，每个用户的历史点击个数是不相等的，我们需要把它们编码成一个固定长的向量。以往的做法是，对每次历史点击做相同的embedding操作之后，将它们做一个求和或者求最大值的操作，类似经过了一个pooling层操作。提出 DIN 的论文认为这个操作损失了大量的信息，于是引入了 attention 机制(其实就是一种加权求和)。

![attention network](https://github.com/Fisher87/ai_explore/blob/master/src/Attunit.png)
DIN模型在对用户的表示计算上引入了attention network (也即图中的Activation Unit) 。DIN把用户特征、用户历史行为特征进行embedding操作，视为对用户兴趣的表示，之后通过attention network，对每个兴趣表示赋予不同的权值。这个权值是由用户的兴趣和待估算的广告进行匹配计算得到的，如此模型结构符合了之前的两个观察——用户兴趣的多样性以及部分对应。

------

#### [7.DCN]()
> 

![DCN模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/DCN.png)

------
