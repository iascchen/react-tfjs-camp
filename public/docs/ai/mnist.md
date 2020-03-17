# 手写数字识别 MNIST

## 数据

MNIST 数据:

* gz 压缩格式的数据，包括 60000 个样本的训练集和测试集数据及其标注：下载地址为 [https://storage.googleapis.com/cvdf-datasets/mnist/](https://storage.googleapis.com/cvdf-datasets/mnist/)
* png 数据，为 Web 版本预处理的图片数据，包括 55000 个数据

## 模型

* 用 LayerModel 构建的多层感知机 MLP 模型
* CNN + 池化 模型
* CNN + Dropout 模型
* 使用 Tensorflow 核心 API 构建的计算网络

## 训练

观察数据，以及随着训练模型参数的变化，观察测试集的推理结果正确情况。

## 推理

在画板上，手写输入数字，观察在其推理输出结果。

## 使用 tfjs-node 加速训练

    ts-node --project tsconfig.node.json ./node/mnist/main.ts
