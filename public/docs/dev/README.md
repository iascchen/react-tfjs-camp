# 开发教程 Develop Tutorial

## 构建开发框架

- [x] [Develop Tutorial 1 从零开始](./start-from-scratch.md)

    * 环境安装
        * 安装 Node 环境
        * 安装 yarn 工具
    * React 和 React Hooks
        * 创建 React 应用
            * 创建一个新的 React 项目
            * React 项目目录简述

- [x] [Develop Tutorial 2 构建 React 开发框架](./dev-structure.md)

    * React-tfjs-camp 的目录结构
        * public 目录结构
        * src 目录结构
        * node 目录结构
    * 规范代码语法和风格检查
        * tsconfig.json
        * .eslintrc.js
    * 改造页面布局
        * React 函数化组件
        * 使用 Ant Design 构建页面框架
            * 在项目中使用 AntD
        * 页面布局
    * 边栏菜单导航
        * AntD Layout Sider
        * 使用 React Hooks 的 useState 管理边栏状态
        * 用 React-Route 实现页面路由跳转
    * ErrorBoundary
    
- [x] [Develop Tutorial 3 搭建展示端到端 AI 概念的舞台](./ai-process-panel.md)

    * 端到端的 AI 概念
    * AIProcessTabs
        * 带参数的 React 函数组件
        * 使用 React Hooks 的 useEffect 处理组件内的数据依赖
        * 处理需要隐藏的 TabPane
        * Sticky 的使用
    * MarkdownWidget

### 操练 Tensorflow.js

- [x] [Develop Tutorial 4 初步了解 Tensorflow.js](./tfjs-intro.md)

    * 使用 Tensorflow.js 的几点须知
        * Backend —— 为什么我的 tfjs 运行很慢？
        * 内存管理 —— 这样避免我的程序内存溢出？
        * tfjs 安装
        * tfjs 加载
    * 使用 Tensorflow.js 和 React 生成数据集
        * 随机生成 a, b, c 三个参数
        * 实现公式计算 & useCallback
        * 训练集和测试集的生成
    * 函数数据可视化
    * 使用 Tensorflow.js 创建人工神经网络
        * 实现一个简单的多层人工神经网络
        * 窥探一下 LayerModel 的内部
    * 模型训练
        * 调整 LearningRate 观察对训练的影响
        * 模型训练 model.fit
        * 及时停止模型训练 —— useRef Hook 登场
    * 模型推理

- [ ] [Develop Tutorial 5 用 Tensorflow.js 处理按数据分类问题](./data-classifier.md)
- [ ] 待续

### 曲线拟合：数学公式线性回归

#### 问题和解决模型

公式拟合和线性回归

##### 神经网络模型

构建一个简单的线性回归神经网络 Model，进行训练和计算

[x] 神经元模型， Dense层次网络模型，tf.Sequential

[x] 问题模型决定神经网络输入输出：Input 和 Output的数据纬度

[x] Loss 和 Optimizer，SGD算法，反向计算

[x] 激活函数：Relu 和 Sigmoid

[x] 训练 model.fit 及其参数

[x] 单层的线性回归，多层模型

#### 构建和运行模型原型

数据样本的生成。

[x] Tensor 基础: 随机数的产生，Tensor的运算，

[x] 数据的展示和便利：Tensor 和 Javascript 数据转换，ES Array：Iterator，Array 的区别和转化, Buffer的使用

[x] 性能优化：Tensor资源的释放(dispose)，tf.tidy GPU资源的释放，数组的释放

#### 构建交互界面，深入理解

[x] React-Hooks: useState, useEffect，useEffect的资源释放，useCallBack

[x] 思考数据数据对界面的触发，确定 useEffect 的数据依赖

[x] 界面交互 AntD，自动生成曲线，驱动训练，曲线的绘制 BizChart，展示训练结果

[x] 探究模型内部信息（Layer 和 Weight）

[x] 修改优化器的超参 LEARNING_RATE，观看超参对训练的影响

### 鸢尾花实验 Iris：推断分类

#### 问题和解决模型

分类推测问题

[ ] 分类问题的模型思路： 标签整数张量，one-hot 编码方式，多分类模型

[ ] 标签整数张量，one-hot 编码方式对 Loss 和 Optimizer 的影响

#### 构建和运行模型原型

[ ] 使用 DataSet / Tensor 作为训练数据的场景辨析

[x] 使用 tf.Data 构建数据集. DataSet(Batch) & tf.data.zip

[x] shape 的使用

[x] one-hot数据的处理

[x] model.fitDataset

[x] 训练回调函数：onEpochEnd 的使用，了解训练的中间状态

#### 构建交互界面，深入理解

[x] one-hot 和 标签整数张量 的选择，展示典型数据

[ ] 变化选择 Loss 和 Optimizer 

[x] 显示训练样本数据，以及Prediction结果

[x] 显示训练状态，BizChart，多条线显示的数据格式，性能优化的考量

### MNIST 手写数字识别：图像分类

#### 问题和解决模型

手写数字识别，分类问题

[x] CNN网络模型，池化和 Dropout

[x] 图片数据的加载和处理：图片数据 / 255

[x] 标签数据的处理：one-hot/标签整数张量

[ ] 数据集的切换：mnist & mnist-fashion

[x] 使用 td-node 执行训练

#### 交互设计和实现

[x] 样本集显示：useRef 和 canvas

[ ] 选择 Dense 网络和 CNN 网络（遗留问题，训练不收敛）

[x] tf-vis 集成，Log 模型处理性能的探讨

[x] 数字手写画板实现

[ ] 卷积层的可视化

[ ] 性能问题，内存优化，tf.dispose(), tf.tidy 等

### MobileNet 图片分类：使用预训练的模型，进行迁移学习

#### 模型

[x] 使用预训练的MobileNet模型. 获得模型和加载 Weights

[x] 使用预训练的MobileNet模型 -> 特征 -> 机器学习算法 KNN [teachable-machine](https://github.com/googlecreativelab/teachable-machine-boilerplate.git)

[x] 使用预训练的MobileNet模型 -> 扩展模型 -> 仅训练靠后扩展的几层 -> 新的可用模型

#### 数据和模型的保存

[x] 模型存储和上传加载

[x] 数据存储和上传加载

#### 交互设计和实现

[x] 图片上传显示组件

[x] 图片分类标注组件

[x] 摄像头组件，拍照上传

[ ] 对视频流的处理

#### 构建一个模型服务器

构建一个 Model Serving 的 Web 服务器（Docker基础）
使用 tfjs.convertor 进行模型转换

### 简单对象识别，基于 Mobilenet 扩展

#### 标注数据的生成

#### 在后台用 Python/Node.js 和 GPU 训练

#### 对象识别

#### 对象计数等

### 处理连续数据：Jena 天气预报

### RNN 文本情感分析 —— RNN和词嵌入

### RNN 文本生成 —— LSTM

### 其他备忘

作诗、作文、作曲Magenta

生成模型：图像风格迁移，

seq-to-seq 的其他应用，DNA 碱基对，代码生成。。。

BERT

使用 RNN 处理视频、音频。。。

### 声音的处理：对信号的AI处理

傅立叶变换将信号转成频谱图 -> 图像处理

声音的输入

基于 LSTM 的 语音识别
