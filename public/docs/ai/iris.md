# 鸢尾花分类 IRIS

## 知识点

这个例子涉及以下的 AI 知识点：

* 分类问题的处理思路：标签整数张量，one-hot 编码方式，多分类模型
* 
构造训练数据
* 多层感知机网络
* 激活函数

## 问题

![梵高鸢尾](../images/iris_02.jpeg)

鸢【音：yuān】尾花（Iris）是单子叶百合目花卉。在北京植物园最北端的宿根花卉区，种植有40余个品种的鸢尾，最佳观赏时间大约在4月下旬至5月中下旬。也就是说，再过一个多月，北京的鸢尾花就应该开了。想必到那时候，新冠疫情应该已经结束，可以和家人朋友一起出去浪，拍几张照片，换换这里网上搜来的照片。

![紫花鸢尾](../images/iris.jpeg)

鸢尾花数据集最初由 Edgar Anderson 测量得到，而后在著名的统计学家和生物学家 R.A Fisher 于 1936 年发表的文章「The use of multiple measurements in taxonomic problems」中被使用，用其作为线性判别分析（Linear Discriminant Analysis）的一个例子，证明分类的统计方法，从此而被众人所知。

鸢尾花数据集由 3 种不同类型的鸢尾花的各 50 个样本数据构成。每个样本包含了4个属性，特征数值都是正浮点数，单位为厘米：

* Sepal.Length（花萼长度）
* Sepal.Width（花萼宽度）
* Petal.Length（花瓣长度）
* Petal.Width（花瓣宽度）

预测变量目标值为鸢尾花的分类为三类，其中的一个种类与另外两个种类是线性可分离的，后两个种类是非线性可分离的：

* Iris Setosa（山鸢尾）
* Iris Versicolour（杂色鸢尾）
* Iris Virginica（维吉尼亚鸢尾）

本节的内容，就是使用测量得到的特征数据，对目标进行分类，是个非常典型的场景。

## 数据

鸢尾花原始的数据，类似这样：

	4.8,3.0,1.4,0.3,Iris-setosa
	5.1,3.8,1.6,0.2,Iris-setosa
	4.6,3.2,1.4,0.2,Iris-setosa
	5.7,3.0,4.2,1.2,Iris-versicolor
	5.7,2.9,4.2,1.3,Iris-versicolor
	6.2,2.9,4.3,1.3,Iris-versicolor
	6.3,3.3,6.0,2.5,Iris-virginica
	5.8,2.7,5.1,1.9,Iris-virginica
	7.1,3.0,5.9,2.1,Iris-virginica

为了便于计算处理，需要对分类结果进行转换处理。

常见的处理分类目标数据的方法有：标签编码 和 One-Hot

### 标签编码

使用 int 类型, 对三种分类进行编号替换，就形成了整数标签目标数据：

* 0 ：Iris setosa（山鸢尾）
* 1 ：Iris versicolor（杂色鸢尾）
* 2 ：Iris virginica（维吉尼亚鸢尾）

上面的数据被转换成（为了便于观察，在数据中增加了空格，以区分特征数据和目标分类数据）：

	4.8,3.0,1.4,0.3, 0
	5.1,3.8,1.6,0.2, 0
	4.6,3.2,1.4,0.2, 0
	5.7,3.0,4.2,1.2, 1
	5.7,2.9,4.2,1.3, 1
	6.2,2.9,4.3,1.3, 1
	6.3,3.3,6.0,2.5, 2
	5.8,2.7,5.1,1.9, 2
	7.1,3.0,5.9,2.1, 2

标签编码的类别值从0开始（因为大多数计算机系统计数），所以，如果有N个类别，类别值为 0 至 N-1 的。

**标签编码的适用场景**：

* 如果原本的标签编码是有序意义的，例如评分等级，使用标签编码就是一个更好的选择。
* 不过，如果标签编码是和鸢尾数据类似的无顺序数据，在计算中，更高的标签数值会给计算带来不必要的附加影响。这时候更好的方案是使用 one-hot 编码方式。

![黑人问号脸](../images/emotion_03.jpg)

在上面的数据中。在进行标签编码的数据集中有

$$ virginica(2) > versicolor(1) > setosa(0) $$

比方说，假设模型内部计算平均值（神经网络中有大量加权平均运算），那么0 + 2 = 2，2 / 2 = 1. 这意味着：virginica 和 setosa 平均一下是 versicolor。该模型的预测会有大量误差。

### One-Hot

One-Hot 编码是将类别变量转换为机器学习算法易于利用的一种形式的过程。One-Hot 将 n 个分类，表示为一个 只含有 0，1 数值的向量。向量的位置表示了对应的分类。

例如，采用 One-Hot 编码，上面的数据就应该编码成：

	4.8,3.0,1.4,0.3, [1,0,0]
	5.1,3.8,1.6,0.2, [1,0,0]
	4.6,3.2,1.4,0.2, [1,0,0]
	5.7,3.0,4.2,1.2, [0,1,0]
	5.7,2.9,4.2,1.3, [0,1,0]
	6.2,2.9,4.3,1.3, [0,1,0]
	6.3,3.3,6.0,2.5, [0,0,1]
	5.8,2.7,5.1,1.9, [0,0,1]
	7.1,3.0,5.9,2.1, [0,0,1]
	
tfjs 里也提供了将标签编码转化成 One-Hot 的函数 `tf.oneHot`，使用起来很方便。

## 模型

鸢尾花的计算模型使用的是两层的全联接网络。

参考代码实现如下。其中激活函数、输入层的神经元数量都可以在页面上直接调整。

	const model = tf.sequential()
	model.add(tf.layers.dense({
		units: sDenseUnits,
		activation: sActivation as any,
		inputShape: [data.IRIS_NUM_FEATURES]
	}))
	model.add(tf.layers.dense({ units: 3, activation: 'Softmax' }))

* 输入层的 inputShape 是和特征数据相关的，是个4元向量。
* 因为要输出三个分类，所以输出层的神经元数量设置为 3。
* 多分类问题的输出层，激活函数使用 Softmax。如果是二分类问题，激活函数可以使用 Sigmoid

### Softmax 激活函数

$$ Softmax(z)_j = \frac{ e^{z_i} }{ \sum e^{z_j} } for i=1..J $$

![Softmax](../images/Softmax.jpg)

Softmax用于多分类神经网络输出. 如果某一个 zj 大过其他 z, 那这个映射的分量就逼近于 1,其他就逼近于 0。

Sigmoid 将一个实数映射到（0,1）的区间，用来做二分类。而 Softmax 把一个 k 维的实数向量（a1,a2,a3,a4…）映射成一个（b1,b2,b3,b4…）其中 bi 是一个 0～1 的常数，输出神经元之和为 1.0，所以相当于概率值，然后可以根据 bi 的概率大小来进行多分类的任务。二分类问题时 Sigmoid 和 Softmax 是一样的，求的都是**交叉墒损失（cross entropy loss）**，而 Softmax 可以用于多分类问题。Softmax 是 Sigmoid的扩展，因为，当类别数 k＝2 时，Softmax 回归退化为 logistic 回归。

## 训练

这次训练除了能够调整 Learning Rate 参数，还能够调整优化算法。

	const optimizer = tf.train.adam(sLearningRate)
	model.compile({ optimizer: sOptimizer, loss: sLoss, metrics: ['accuracy'] })

### Loss 函数的选择

Loss 函数对于训练非常重要。

在这个例子里，根据目标数据编码形式的不同，需要选用不同的 Loss 函数。

* 标签编码: sparseCategoricalCrossentropy
* One-Hot: categoricalCrossentropy
 
### 优化器算法

#### SGD
这是最基础的梯度下降算法，更新权重W，不多解释。

![SGD](../images/sgd_01.jpeg)

其中 α是learning rate(学习速率)。我们可以把下降的损失函数看成一个机器人，由于在下降的时候坡度不是均匀的，机器人会左右摇摆，所以下降速度会比较慢，有时候遇到局部最优，还可能在原地徘徊好长时间。

![SGD](../images/sgd.png)

#### RMSProp

RMSprop 是 Geoff Hinton 提出的一种自适应学习率方法。Hinton 建议设定 γ 为 0.9, 学习率 η 为 0.001。

![RMSProp](../images/rmsprop_01.jpeg)

#### Adam

Adam是目前用得最广的优化算法。这个算法是一种计算每个参数的自适应学习率的方法。和 RMSprop 一样存储了过去梯度的平方 vt 的指数衰减平均值 ，也保持了过去梯度 mt 的指数衰减平均值。相当于给机器人一个惯性，同时还让它穿上了防止侧滑的鞋子，当然就相当好用啦。

建议 β1 ＝ 0.9，β2 ＝ 0.999，ϵ ＝ 10e−8。实践表明，Adam 比其他适应性学习方法效果要好。

![Adam](../images/adam_01.jpeg)
![Adam](../images/adam_02.jpeg)
![Adam](../images/adam_03.jpeg)

#### 不同优化算法下降速度的差距

![optimizer_02](../images/optimizer_02.gif)

![optimizer](../images/optimizer.gif)

## 扩展阅读

### 鸢尾花数据集

[鸢尾花数据集](https://www.jianshu.com/p/6ada344f91ce)

### One-Hot

[什么是one hot编码？为什么要使用one hot编码？](https://zhuanlan.zhihu.com/p/37471802)

### 优化器

[关于深度学习优化器 optimizer 的选择，你需要了解这些](https://www.leiphone.com/news/201706/e0PuNeEzaXWsMPZX.html)

[Tensorflow中的Optimizer(优化器)](https://www.jianshu.com/p/8f9247bc6a9a)
