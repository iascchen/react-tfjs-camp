# 处理连续数据的模型：循环神经网络

## Jena 天气预报

### 数据

**注意** 

* 如果您要在本地环境运行这个例子，最好预先下载数据文件。并将数据文件放在此项目的 `./public/data` 目录下。

    [https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv](https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv)

* 所需的数据大约有 41.2MB。
* 刷新页面，会丢失已经加载的数据。

这个例子里最重要的部分是构建训练数据集的部分，参考如下相关代码。

    getNextBatchFunction = (shuffle: boolean, lookBack: number, delay: number, batchSize: number, step: number, minIndex: number, maxIndex: number, normalize: boolean,
        includeDateTime: boolean): any => {
        let startIndex = minIndex + lookBack
        const lookBackSlices = Math.floor(lookBack / step)

        return {
            next: () => {
                const rowIndices = []
                let done = false // Indicates whether the dataset has ended.
                if (shuffle) {
                    // If `shuffle` is `true`, start from randomly chosen rows.
                    const range = maxIndex - (minIndex + lookBack)
                    for (let i = 0; i < batchSize; ++i) {
                        const row = minIndex + lookBack + Math.floor(Math.random() * range)
                        rowIndices.push(row)
                    }
                } else {
                    // If `shuffle` is `false`, the starting row indices will be sequential.
                    let r = startIndex
                    for (; r < startIndex + batchSize && r < maxIndex; ++r) {
                        rowIndices.push(r)
                    }
                    if (r >= maxIndex) {
                        done = true
                    }
                }

                const numExamples = rowIndices.length
                startIndex += numExamples

                const featureLength =
                    includeDateTime ? this.numColumns + 2 : this.numColumns
                const samples = tf.buffer([numExamples, lookBackSlices, featureLength])
                const targets = tf.buffer([numExamples, 1])
                // Iterate over examples. Each example contains a number of rows.
                for (let j = 0; j < numExamples; ++j) {
                    const rowIndex = rowIndices[j]
                    let exampleRow = 0
                    // Iterate over rows in the example.
                    for (let r = rowIndex - lookBack; r < rowIndex; r += step) {
                        let exampleCol = 0
                        // Iterate over features in the row.
                        for (let n = 0; n < featureLength; ++n) {
                            let value
                            if (n < this.numColumns) {
                                value = normalize ? this.normalizedData[r][n] : this.data[r][n]
                            } else if (n === this.numColumns) {
                                // Normalized day-of-the-year feature.
                                value = this.normalizedDayOfYear[r]
                            } else {
                                // Normalized time-of-the-day feature.
                                value = this.normalizedTimeOfDay[r]
                            }
                            samples.set(value, j, exampleRow, exampleCol++)
                        }

                        const value = normalize
                            ? this.normalizedData[r + delay][this.tempCol]
                            : this.data[r + delay][this.tempCol]
                        targets.set(value, j, 0)
                        exampleRow++
                    }
                }
                return {
                    value: { xs: samples.toTensor(), ys: targets.toTensor() },
                    done
                }
            }
        }
    }

### 模型

提供以下神经网络模型，供比较。

* linear-regression 单层线性回归模型
* mlp 多层感知机
* mlp-l2 多层感知机，使用 L2 正则化
* mlp-dropout 多层感知机，使用 Dropout 处理过拟合
* simpleRnn 简单的 RNN 模型
* gru GRU 模型

https://zhuanlan.zhihu.com/p/32481747

模型的输入 Shappe 和所加载的特征数据的列数有关。(=14）
    
#### RNN

循环神经网络（Recurrent Neural Network，RNN）是一种用于处理序列数据的神经网络。
相比一般的神经网络来说，它能够处理序列变化的数据。比如某个单词的意思会因为上文提到的内容不同而有不同的含义，RNN就能够很好地解决这类问题。

https://zhuanlan.zhihu.com/p/32085405

RNN 的伪代码

        y=0
        for x in input_sequence:
            y = f(dot(W, x) + dot(U, y))

#### GRU

GRU（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network, RNN）的一种。和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。

GRU和LSTM在很多情况下实际表现上相差无几，那么为什么我们要使用新人GRU（2014年提出）而不是相对经受了更多考验的LSTM（1997提出）呢。
"我们在我们的实验中选择GRU是因为它的实验效果与LSTM相似，但是更易于计算。"
简单来说就是贫穷限制了我们的计算能力...

相比LSTM，使用GRU能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。

GRU 的伪代码

        h=0
        for x_i in input_sequence:
            z = sigmoid(dot(W_z, x) + dot(U_z, h))
            r = sigmoid(dot(W_r, x) + dot(W_r, h)) 
            h_prime = tanh(dot(W, x) + dot(r, dot(U, h))) 
            h = dot(1 - z, h) + dot(z, h_prime)

#### LSTM

https://zhuanlan.zhihu.com/p/74034891

### 使用 RNN 进行语义分析

对文章进行 multi-hot 分析，对 IMDB 数据进行分析。

更高效的技术：词嵌入



我们已经了解了如何使用和构建自己的 CNN 网络。
接下来，我们来认识另外一个重要的深度神经网络模型：循环神经网络 RNN
RNN 被用于处理文本理解、语音识别等场景。这类场景的共同特点是：

**问题结果与过去的几个输入数据都相关**

简单说来就是 y = f( Wx + Uy)



RNN



LSTM

中文分词

作诗、作文、作曲Magenta

生成模型：图像风格迁移，

seq-to-seq 的其他应用，DNA 碱基对，代码生成。。。

BERT

使用 RNN 处理视频、音频。。。

## 第九章 声音的处理：对信号的AI处理

傅立叶变换将信号转成频谱图 -> 图像处理

声音的输入

基于 LSTM 的 语音识别

## GAN 对抗生成网络
