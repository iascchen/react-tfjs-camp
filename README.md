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

安装 Node 环境

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash
    
    
    curl -o- -L https://yarnpkg.com/install.sh | bash
    
    wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash

    
> In China, you can use taobao npm registry

    npm config set registry https://registry.npm.taobao.org
    
### 创建项目    
    
    npx create-react-app react-tfjs-playground --template typescript
    cd react-tfjs-playground
    yarn
    yarn start
    
这是一个经典的 React 启动项目。你可以尝试一下一下的命令：

    yarn test
    yarn build

将项目发布到 Web 上

    yarn eject
    
### 构建程序框架

简介 React-Hooks， React-Route，ESLint，tsconfig.json。

使用 AntD 构造了一个左右结构的程序框架
    
## 第三章：数学公式线性回归：曲线拟合拟合

### 数据

Tensor 基础，

* 数据的生成。曲线生成。。。
* ES Array：Iterator，Array 的区别和转化
* Buffer 的使用

### 模型
构建一个简单的线性回归神经网络 Model，显示 Model 的信息

* 基本网络：tf.Sequriel
* Operator 和 loss
* RELU 和 Sigmod
* 单层和多层

### 界面

React 的交互界面（简述）和 React-Hooks 使用的技巧

useEffect 的 注意事项 

* ANTD 的使用
* ANTV 的 图表生成
* React Hooks 深入：dispose 过程， useCallBack，useEffect
* 数据联动（undefined 的处理）

训练和推理

## 第四章：鸢尾花实验 Iris：推断分类
推测分类

### 数据

使用 tf.Data 构建数据集

* DataSet tf.data.zip
* one-hot 编码方式 
* shape 的使用
* mul-hot

### 模型

onEpochEnd 的使用

### 界面

显示训练状态，AntV
对比训练结果
    
## 第四章：MNIST 手写数字识别：图像识别

### 数据

* 数据的加载和解析
* 图片数据的处理 / 255

### 网络

CNN网络模型
卷基层的可视化

### 显示

* useRef 和 canvas
（在React中处理 Canvas ：手写数字，显示分割图片）
* 内存的优化 tidy 等
* Log 模型处理的优化，tf-vis/tensor-board
* 数字手写板
* 摄像头识别。绑定摄像头

## 第五章：MobileNet 图片分类：使用预训练的模型，进行迁移学习

### 模型

MobileNet，PoseNet 模型

* 使用预训练的模型
* 迁移学习
* tf.LayerModel 扩展自己的个性化网络
* 模型存储和上传加载

特征的提取 => 人脸特征点

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


