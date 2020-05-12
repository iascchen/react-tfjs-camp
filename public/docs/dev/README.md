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

- [x] [Develop Tutorial 5 用 Tensorflow.js 处理按数据分类问题](./data-classifier.md)

    * 分类问题的数据表述
        * 标签编码
        * One-Hot
    * 用 tf.data.Dataset 构造训练集和测试集
        * 按比例分配数据集
        * 了解 tf.data.Dataset
        * 初始化数据集
        * SampleDataVis 展示数据样本
            * 使用 useEffect 构建细粒度的数据驱动渲染
            * AntD Table 的使用
    * 全联接网络模型
    * 训练
        * 调整训练参数：注意一下 Loss 函数
        * 使用 Model.fitDataset 训练
        * 展示训练过程 —— 在 useState 中使用数组
    
- [x] [Develop Tutorial 6 MNIST CNN 的 Layer Model 实现](./mnist-layer-model.md)

    * MNIST 的数据集
        * MNIST 的数据集的两种格式—— PNG 和 GZ
        * 预先下载数据集到本地
        * PNG 格式数据的加载和使用
        * GZ 格式数据的加载和使用
            * 使用 fetch 加载数据文件
            * 数据的加载
    * 修改 SampleDataVis 以显示图片
        * 组件 RowImageWidget—— 使用 useRef 访问 HTML Element
    * CNN 网络模型
        * 将 tfjs-vis 集成到 React
    * 模型训练
    * 推理
        * 数字手写板的实现 —— 在 React 中使用 canvas 绘图
        * 使用 Tfjs 将 canvas 位图转化为 Tensor

- [ ] [Develop Tutorial 7 MNIST CNN 的 Core API 实现](./mnist-core-api.md)

- [ ] 待续

===========================


[ ] 使用 td-node 执行训练

[ ] 卷积层的可视化


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
