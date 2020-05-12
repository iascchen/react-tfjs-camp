# MNIST CNN 的 Core API 实现

## 数据分片加载

下面的代码引用自 `./src/componenets/mnist/MnistDatasetPng.ts`。使用 `tf.util.createShuffledIndices` 来实现按批次随机抽取数据分片的目的。

	    loadData = async (): Promise<void> => {
			...
			// Create shuffled indices into the train/test set for when we select a
	        // random dataset element for training / validation.
	        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
	        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS)
	        ...
	    }

		nextTrainBatch = (batchSize: number): tf.TensorContainerObject => {
	        return this.nextBatch(batchSize, [this.trainImages, this.trainLabels],
	            () => {
	                this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
	                return this.trainIndices[this.shuffledTrainIndex]
	            })
	    }
	    
	    ...
	    
	    nextBatch = (batchSize: number, data: [Float32Array, Uint8Array], index: Function): tf.TensorContainerObject => {
	        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE)
	        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)
	
	        for (let i = 0; i < batchSize; i++) {
	            const idx = index() as number
	
	            const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE)
	            batchImagesArray.set(image, i * IMAGE_SIZE)
	
	            const label = data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES)
	            batchLabelsArray.set(label, i * NUM_CLASSES)
	        }
	
	        const xs = tf.tensor4d(batchImagesArray, [batchSize, IMAGE_H, IMAGE_W, 1])
	        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])
	
	        return { xs, ys: labels }
	    }

## 使用 Tensorflow.js 的 Core API 构造深度神经网络

相关代码在 `./src/componenets/mnist/modelCoreApi.ts`。

### 等价的 Layers API 实现

本节采用 Core API 实现的 CNN 模型，等价于下面这段 Layers API 所描述的网络。
 
 	const model = tf.sequential()
 	
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1], kernelSize: 5, filters: 8, activation: 'relu', padding: 'same'
    }))
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.conv2d({ kernelSize: 5, filters: 16, activation: 'relu', padding: 'same'}))
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.flatten({}))
    
	model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))

### 卷积模型的权重参数

	// Variables that we want to optimize
	const conv1OutputDepth = 8
	const conv1Weights = tf.variable(tf.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1))
	
	const conv2InputDepth = conv1OutputDepth
	const conv2OutputDepth = 16
	const conv2Weights = tf.variable(tf.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1))

	const fullyConnectedWeights = tf.variable(
	    tf.randomNormal([7 * 7 * conv2OutputDepth, NUM_CLASSES], 0,
	        1 / Math.sqrt(7 * 7 * conv2OutputDepth)))
	const fullyConnectedBias = tf.variable(tf.zeros([NUM_CLASSES]))
	
* conv1Weights 的形状 [5, 5, 1, conv1OutputDepth] 所对应的维度含义是  [filterHeight, filterWidth, inDepth, outDepth]，描述了此卷基使用 5 * 5 的卷积核，输入数据深度为 1，输出数据深度为 8。
* conv2Weights 的形状 [5, 5, conv1OutputDepth, conv2OutputDepth]，描述了此卷基使用 5 * 5 的卷积核，输入数据深度为 8，输出数据深度为 16。
* 输出层计算时，将 layer2 的输出，被扁平化为长度 784 的一维向量，输出长度为 10 的结果向量。

### 卷积模型的前向传播计算过程

接下来，我们看看输入一组训练数据 xs 后，在模型中是如何计算的。

	export const model = (inputXs: tf.Tensor): tf.Tensor => {
	    const xs = inputXs.as4D(-1, IMAGE_H, IMAGE_W, 1)
	
	    const strides = 2
	    const pad = 0
	
	    // Conv 1
	    const layer1 = tf.tidy(() => {
	        return xs.conv2d(conv1Weights as tf.Tensor4D, 1, 'same')
	            .relu()
	            .maxPool([2, 2], strides, pad)
	    })
	
	    // Conv 2
	    const layer2 = tf.tidy(() => {
	        return layer1.conv2d(conv2Weights as tf.Tensor4D, 1, 'same')
	            .relu()
	            .maxPool([2, 2], strides, pad)
	    })
	
	    // Final layer
	    return layer2.as2D(-1, fullyConnectedWeights.shape[0])
	        .matMul(fullyConnectedWeights as tf.Tensor)
	        .add(fullyConnectedBias)
	}
 
1. 输入 Tensor4D 为形状为 [-1, 28, 28, 1] 的 xs 数据。shape[0] 为 `-1` 表示使用 Tensor 在这个维度的实际值。
2. 经过 layer1 计算。先做了卷积核为 5 * 5 的卷积核，`'same'` 参数表示，卷积后输出数据的 shape 保持与输入一致, 输出形状为 [-1, 28, 28, 8]。
3. 使用 relu 做激活。
4. 使用 maxPool 进行 [2,2] 池化。输出形状为 [-1, 14, 14, 8]。
5. 经过 layer2 计算。输出形状为 [-1, 14, 14, 16]。
6. 使用 relu 做激活。
7. 使用 maxPool 进行 [2,2] 池化。输出形状为 [-1, 7, 7, 16]
8. Final 层计算，首先将 [-1, 7, 7, 16] 的数据扁平化为 [-1, 784] 的向量表达
9. 经过 matMul 和 add 计算后，形成 [-1, 10] 的输出 One-Hot 结果。

### 模型的训练——被隐藏的梯度下降和反向传播

选用对输出的 One-Hot 结果经过 softmax 计算后的交叉熵为 Loss 值。

	// Loss function
	const loss = (labels: tf.Tensor, ys: tf.Tensor): tf.Scalar => {
	    return tf.losses.softmaxCrossEntropy(labels, ys).mean()
	}

Tensorflow 对于模型的自动求导是靠各式各样的 Optimizer 类进行的，我们只需要在程序中构建前向图，然后加上Optimizer，再调用minimize()方法就可以完成梯度的反向传播。

	// Train the model.
	export const train = async (data: IMnistDataSet, log: Function,
	    steps: number, batchSize: number, learningRate: number): Promise<void> => {
	    const returnCost = true
	    const optimizer = tf.train.adam(learningRate)
	
	    for (let i = 0; i < steps; i++) {
	        const cost = optimizer.minimize(() => {
	            const batch = data.nextTrainBatch(batchSize)
	            const _labels = batch.ys as tf.Tensor
	            const _xs = batch.xs as tf.Tensor
	            return loss(_labels, model(_xs))
	        }, returnCost)
	
	        log(i, cost?.dataSync())
	        await tf.nextFrame()
	    }
	}
	
Optimizer class是所有 Optimizer 的基类，整个反向传播过程可分为三步，这三步仅需通过一个minimize()函数完成：

1. 计算每一个部分的梯度，compute_gradients()
2. 根据需要对梯度进行处理
3. 把梯度更新到参数上，apply_gradients()

