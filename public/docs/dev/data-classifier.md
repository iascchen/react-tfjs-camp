# 用 Tensorflow.js 处理按数据分类问题

## 分类问题的数据表述

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

标签编码的类别值从0开始（因为大多数计算机系统如此计数），所以，如果有N个类别，类别值为 0 至 N-1 的。

鸢尾花数据量不大，直接按照整数标签编码，在代码中定义为一个多维数组。

	export const IRIS_RAW_DATA = [
	    [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0],
	    [4.6, 3.1, 1.5, 0.2, 0], [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0],
	    [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0], [4.4, 2.9, 1.4, 0.2, 0],
	    ...
	    [6.9, 3.1, 5.1, 2.3, 2], [5.8, 2.7, 5.1, 1.9, 2], [6.8, 3.2, 5.9, 2.3, 2],
	    [6.7, 3.3, 5.7, 2.5, 2], [6.7, 3.0, 5.2, 2.3, 2], [6.3, 2.5, 5.0, 1.9, 2],
	    [6.5, 3.0, 5.2, 2.0, 2], [6.2, 3.4, 5.4, 2.3, 2], [5.9, 3.0, 5.1, 1.8, 2]
	]

**标签编码的适用场景**：

* 如果原本的标签编码是有序意义的，例如评分等级，使用标签编码就是一个更好的选择。
* 不过，如果标签编码是和鸢尾数据类似的无顺序数据，在计算中，更高的标签数值会给计算带来不必要的附加影响。这时候更好的方案是使用 one-hot 编码方式。

在上面的数据中。在进行标签编码的数据集中有

$$ virginica(2) > versicolor(1) > setosa(0) $$

比方说，假设模型内部计算平均值（神经网络中有大量加权平均运算），那么0 + 2 = 2，2 / 2 = 1. 这意味着：virginica 和 setosa 平均一下是 versicolor。如果不对Loss 函数作些变化，该模型的预测也许会有大量误差。

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

## 构造训练集和测试集

### 按比例分配数据集

	export const splitDataSet = (shuffled: IArray, testSplit: number, shuffle = false): IArray[] => {
	    if (shuffle) {
	        tf.util.shuffle(shuffled)
	    }
	
	    const totalRecord = shuffled.length
	    // Split the data into training and testing portions.
	    const numTestExamples = Math.round(totalRecord * testSplit)
	    const numTrainExamples = totalRecord - numTestExamples
	
	    const train = shuffled.slice(0, numTrainExamples)
	    const test = shuffled.slice(numTrainExamples)
	
	    return [train, test]
	}

* 利用 ES6 的数组函数 `Array.slice()` 简单粗暴的按照数组下标将原始数据分拆成训练集和测试集。
* `tf.util.shuffle` 被用于打乱数组数据的顺序。这是在做数据处理时经常用到的方法。

> ES6 里还有个很形似 `Array.slice()` 的函数，`Array.splice()`，不小心的话容易混淆，需要注意区分一下。
> 
> * slice()方法返回数组中被选中的元素，作为一个新的数组对象。splice()方法返回数组中被删除的项。
* slice()方法不改变原来的数组，而splice()方法改变了原来的数组。
* slice()方法可以接受2个参数。splice()方法可以接受n个参数。

### 了解 tf.data.Dataset

有两种方法可以训练LayersModel ：

* 使用 `model.fit()` 并将数据作为一个大张量提供。
* 使用 `model.fitDataset()` 并通过 Dataset 对象提供数据.

在 Curve 的例子中，我们已经使用 Tensor 作为数据，对模型进行了训练。如果您的数据集能够被放进内存，并且可以作为单个张量使用，则可以通过调用 fit() 方法来训练模型。

而如果数据不能完全放入内存或正在流式传输，则可以通过使用 Dataset 对象的 fitDataset() 来训练模型. 

Dataset 表示一个有序的元素集合对象，这个对象能够通过链式方法完成一系列加载和转换，返回另一个 Dataset。数据加载和转换是以一种懒加载和流的方式完成。数据集可能会被迭代多次；并且每次迭代都会从头开始进行。例如：

	const processedDataset = rawDataset.filter(...).map(...).batch(...)
	
下面的代码被用于生成鸢尾花的 DataSet，来自 `./src/components/iris/data.ts`。

	export const getIrisData = (testSplit: number, isOntHot = true,
	    shuffle = true): Array<tf.data.Dataset<tf.TensorContainer>> => {
	    // Shuffle a copy of the raw data.
	    const shuffled = IRIS_RAW_DATA.slice()
	    const [train, test] = splitDataSet(shuffled, testSplit, shuffle)
	
	    // Split the data into into X & y and apply feature mapping transformations
	    const trainX = tf.data.array(train.map(r => r.slice(0, 4)))
	    const testX = tf.data.array(test.map(r => r.slice(0, 4)))
	
	    let trainY: tf.data.Dataset<number[]>
	    let testY: tf.data.Dataset<number[]>
	
	    if (isOntHot) {
	        trainY = tf.data.array(train.map(r => flatOneHot(r[4])))
	        testY = tf.data.array(test.map(r => flatOneHot(r[4])))
	    } else {
	        trainY = tf.data.array(train.map(r => [r[4]]))
	        testY = tf.data.array(test.map(r => [r[4]]))
	    }
	
	    // Recombine the X and y portions of the data.
	    const trainDataset = tf.data.zip({ xs: trainX, ys: trainY })
	    const testDataset = tf.data.zip({ xs: testX, ys: testY })
	
	    return [trainDataset, testDataset]
	}

* 将每条鸢尾花数据的前四个元素作为 X。

	    // Split the data into into X & y and apply feature mapping transformations
	    const trainX = tf.data.array(train.map(r => r.slice(0, 4)))
	    const testX = tf.data.array(test.map(r => r.slice(0, 4)))

* Y 则根据所使用的编码方式而发生变化。

		if (isOntHot) {
	        trainY = tf.data.array(train.map(r => flatOneHot(r[4])))
	        testY = tf.data.array(test.map(r => flatOneHot(r[4])))
	    } else {
	        trainY = tf.data.array(train.map(r => [r[4]]))
	        testY = tf.data.array(test.map(r => [r[4]]))
	    }

* 将整数标签编码转换成 OneHot 编码的函数如下。

		export const flatOneHot = (idx: number): number[] => {
		    return Array.from(tf.oneHot([idx], 3).dataSync())
		}

### 初始化数据集

我们需要根据用户选择的 sTargetEncode 更新训练和测试数据集，这样的代码，放在 useEffect 里很适合，来自 `./src/components/iris/Iris.tsx`。

    useEffect(() => {
        if (!sTargetEncode) {
            return
        }
        logger('encode dataset ...')

        const isOneHot = sTargetEncode === ONE_HOT
        const [tSet, vSet] = data.getIrisData(VALIDATE_SPLIT, isOneHot)

        // Batch datasets.
        setTrainSet(tSet.batch(BATCH_SIZE))
        setValidSet(vSet.batch(BATCH_SIZE))
        
        ...
    }, [sTargetEncode])

* 这里有个前面强调过的知识点，Dataset.batch 会在最后训练发生时，才会去迭代执行，产生数据片段。

### SampleDataVis 展示数据样本

为了便于观察数据样本，构造了 SampleDataVis 组件，来自 `./src/components/common/tensor/SampleDataVis.tsx`。

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

* 输入层的 inputShape 是和特征数据相关的，是个 4 元向量。
* 因为要输出三个分类，所以输出层的神经元数量设置为 3。
* 多分类问题的输出层，激活函数使用 Softmax。如果是二分类问题，激活函数可以使用 Sigmoid

