#### [1.DeepCrossing]()

------

#### [2.FNN(Factorisation-machine supported Neural Networks)]()

> [paper](https://arxiv.org/pdf/1601.02376.pdf)

![FNN模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/FNN.png)

    `FNN`假设输入数据的格式是离散的类别特征(表示为`one-hot`编码)，且**每个特征属于一个`field`**，通过`embedding层`将高纬稀疏特征映射成低维稠密特征后，再作为多层感知机(MLP)的输入。
一般来说，embedding 层的参数可以随机初始化，**但是在FNN中，初始化embedding是采用通过FM预训练得到的每个特征的隐向量**，这样初始化的好处是将预训练的向量作为初始化参数时，能够让模型的参数在初始化的时候就处于较优的位置(训练的目的其实就是为了得到最优的模型参数)，能够加快收敛的过程，至于效果方面，则不一定会优于随机初始化的情况，因为随机初始化经过多轮的迭代也可能会收敛同样的效果。

------

#### [3.PNN]()

> [paper](https://arxiv.org/pdf/1611.00144.pdf)

![PNN模型框架](https://github.com/Fisher87/ai_explore/blob/master/src/PNN.png)

    `PNN`是在`FNN`的基础上进行改进，就是**增加了特征的二阶交叉项**. 因此, `FNN`和`PNN`的关系，类似于`LR`和`FM`的关系，只是`FNN`和`PNN`均是对原始特征进行了`embedding`映射;

------


#### [4.WideDeep]()

------

#### [5.DeepFM]()

------

#### [6.DIN]()

------

#### [7.DCN]()

------
