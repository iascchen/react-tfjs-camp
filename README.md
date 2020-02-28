# react-tfjs-playground 全栈 AI 训练

## 第一章：概述

这是一个以学习 tensorflow.js 为目的的项目。
使用 React Hooks + Typescript 实现主要界面。

### 技术栈

* Typescript (v3.7.2)
* React (v16.12) : react-scripts, react-hooks, react-router
* Tensorflow.js (v1.5.2)
* AntD & AntV/bizcharts

### 快速开始

    git clone https://github.com/iascchen/react-tfjs-playground.git
    yarn
    yarn start
    
Open [http://loalhost:3000](http://loalhost:3000)

## 第二章：从零开始 Start from scratch

### 环境安装

安装 Node环境

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash    
    nvm install 13
    node --version

安装 yarn 工具
    
    curl -o- -L https://yarnpkg.com/install.sh | bash
    yarn --version
 
> In China, you can use taobao npm registry

    npm config set registry https://registry.npm.taobao.org
    
### 创建项目    
    
    npx create-react-app react-tfjs-playground --template typescript
    cd react-tfjs-playground
    yarn
    yarn start
    
这是一个经典的 React 启动项目。你可以尝试一下其它的命令：

    yarn test
    yarn build
    
### 构建程序框架

[x] React 概述

[x] 简介 React-Hooks， React-Route

[x] 代码风格检查 ESLint，tsconfig.json。

[x] 使用 AntD 构造了一个左右结构的程序框架

[ ] MD 文档显示Widget
    
## 第三章：数学公式线性回归：曲线拟合

### 问题和解决模型

公式拟合和线性回归

#### 神经网络模型

构建一个简单的线性回归神经网络 Model，进行训练和计算

[x] 神经元模型， Dense层次网络模型，tf.Sequential

[x] 问题模型决定神经网络输入输出：Input 和 Output的数据纬度

[x] Loss 和 Optimizer，SGD算法，反向计算

[x] 激活函数：Relu 和 Sigmoid

[x] 训练 model.fit 及其参数

[x] 单层的线性回归，多层模型

### 构建和运行模型原型

数据样本的生成。

[x] Tensor 基础: 随机数的产生，Tensor的运算，

[x] 数据的展示和便利：Tensor 和 Javascript 数据转换，ES Array：Iterator，Array 的区别和转化, Buffer的使用

[x] 性能优化：Tensor资源的释放(dispose)，tf.tidy GPU资源的释放，数组的释放

### 构建交互界面，深入理解

[x] React-Hooks: useState, useEffect，useEffect的资源释放，useCallBack

[x] 思考数据数据对界面的触发，确定 useEffect 的数据依赖

[x] 界面交互 AntD，自动生成曲线，驱动训练，曲线的绘制 BizChart，展示训练结果

[x] 探究模型内部信息（Layer 和 Weight）

[x] 修改优化器的超参 LEARNING_RATE，观看超参对训练的影响

## 第四章：鸢尾花实验 Iris：推断分类

### 问题和解决模型

分类推测问题

[ ] 分类问题的模型思路： 标签整数张量，one-hot 编码方式，多分类模型

[ ] 标签整数张量，one-hot 编码方式对 Loss 和 Optimizer 的影响

### 构建和运行模型原型

[ ] 使用 DataSet / Tensor 作为训练数据的场景辨析

[x] 使用 tf.Data 构建数据集. DataSet(Batch) & tf.data.zip

[x] shape 的使用

[x] one-hot数据的处理

[x] model.fitDataset

[x] 训练回调函数：onEpochEnd 的使用，了解训练的中间状态

### 构建交互界面，深入理解

[x] one-hot 和 标签整数张量 的选择，展示典型数据

[ ] 变化选择 Loss 和 Optimizer 

[x] 显示训练样本数据，以及Prediction结果

[x] 显示训练状态，BizChart，多条线显示的数据格式，性能优化的考量

## 第四章：MNIST 手写数字识别：图像分类

### 问题和解决模型

手写数字识别，分类问题

[x] CNN网络模型，池化和 Dropout

[x] 图片数据的加载和处理：图片数据 / 255

[x] 标签数据的处理：one-hot/标签整数张量

[ ] 数据集的切换：mnist & mnist-fashion

### 交互设计和实现

[x] 样本集显示：useRef 和 canvas

[ ] 选择 Dense 网络和 CNN 网络（遗留问题，训练不收敛）

[x] tf-vis 集成，Log 模型处理性能的探讨

[x] 数字手写画板实现

[ ] 卷积层的可视化

[ ] 性能问题，内存优化，tf.dispose(), tf.tidy 等

## 第五章：MobileNet 图片分类：使用预训练的模型，进行迁移学习

### 模型

[x] 使用预训练的MobileNet模型. 获得模型和加载 Weights

[x] 使用预训练的MobileNet模型 -> 特征 -> 机器学习算法 KNN [teachable-machine](https://github.com/googlecreativelab/teachable-machine-boilerplate.git)

[ ] 使用预训练的MobileNet模型 -> 扩展模型 -> 仅训练靠后扩展的几层 -> 新的可用模型

特征的提取 => 人脸特征点

### 数据和模型的保存

[x] 模型存储和上传加载

[x] 数据存储和上传加载

### 交互设计和实现

[x] 图片上传显示组件

[x] 图片分类标注组件

[x] 摄像头组件，拍照上传、对视频流的处理

### 构建一个模型服务器

构建一个 Model Serving 的 Web 服务器（Docker基础）
使用 tfjs.convertor 进行模型转换

## 第七章：Mobile-SSD 对象识别：使用后台计算

数据标注

构建一个标注数据的上传服务器

在后台用 Python/Node.js 和 GPU 训练

对象识别

对象计数

利用 Yolo 对视频进行对象识别

### 数据标注



## 第八章：RNN 文本生成：自然语言处理

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

## 第十章 对话机器人

翻译

## 第十一章 强化学习：玩游戏

metacar-project.com

## 第十二章 遗产算法赛车：玩游戏

## 第十三章 博弈：AlphoGO 

## 第十四章 联邦学习

## 测试和优化


