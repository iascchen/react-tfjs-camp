# react-tfjs-camp 全栈 AI 训练

## 缘起

多灾多难的 2020 年，必定会在历史上记一笔。春节以来，COVID-19 疫情日益严重，全国人民都在家隔离。憋在家里的这段日子，每天被各种信息坏消息和好消息蹂躏来，蹂躏去。
时间久了，也渐渐应激疲劳，静下来，找点事，排遣一下时间。

最近几年，我一直在寻找如何让用户拥有通过私有数据训练，实现 “私有AI” 能力的途径。

我们先来看看当前的经典实现架构：

[前端AI应用+模型Serving+后端计算的示意图]

你的所有数据都会需要传送到后端服务器，再使用服务端资源进行训练。一来数据传输量大，二来服务端计算资源要求也不老少。这个模式提高了小型AI企业创新服务的门槛，没有个几百上千台服务器，你都不好意思提供 SaaS 服务。

相对理想的架构是将更多的计算放在客户端，也就是大家常说“边缘计算”。服务端负责提供在“大数据”上预先训练好的基础模型，客户端在本地上进行“小数据”个性化训练，实现“私有 AI”。

### Tensorflow.js

Tensorflow.js 在 2月4日 除夕那天发布了 v1.5.2，能够在浏览器端做训练了！！！能直接在浏览器端做训练，即使只是“简单”的训练，是个很有意义的大进步。可以在用户（浏览器）端进行个性化的训练，进行隐私信息的脱敏，再和服务端更强大的AI能力相配合，相信不久之后，我们能够看到更多既能保持个人数据隐私，也足够智能灵活的新型AI应用。Google也提供了“联邦计算”的例子，也许这是“私人AI”时代的曙光。

[后端预训练+前端个性化训练+联邦计算的示意图]

除此之外，模型加载、存储和转换相关功能基本完备，能够很顺畅地与 Tensorflow v2.0 以上版本训练所产生的模型进行对接。

### React Hooks

因为工作性质的关系，我自己的主要时间都放在了团队管理、技术预研和架构设计方面，虽然也动手做一部分代码，但是对于不少新的技术特性细节，都得靠团队小伙伴们的鼎力支持了。

我的团队主要使用基于 JavaScript/TypeScript 的 React + Redux + Express 进行全栈开发，除了人工智能后端用 Python。

使用 React 已经是相当舒适的编程体验了，不过如果涉及到复杂的组件间信息交互，以及多层父子之间的数据共享，就必须使用高阶组件+Context 或者 Connect 传递数据，这样的代码风格，不够简洁和优美。

React Hooks 是 React 在 16.8 之后支持的新的特性，发布至今已经一年了吧。因为当前工作中的项目里一直使用的是 Component 的对象形式，并没有太仔细的学习和使用。React Hooks 更容易将组件的 UI 与状态分离，不用再写很看起来不爽的 setState，不用担心在 componentDidMount、componentWillUnmount 忘了做这做那。使用 React Hooks 开发，可以用很直观的数据变化驱动界面变化的思路来组织代码，实际使用，还真香。

### TypeScript

工作中，只要求团队在后端开发使用 TypeScript。

作为一个资深老 Java，最近几年，使用 JavaScript 和 Python 居多，确实有一种放飞自我的感觉。源自于 Java 开发养成的代码习惯，加上 JS 和 Python 的灵活方便，大大提高了我将创意和想法快速开发实现的效率。不过，因为经常在多种语言之间切来切去，不少语法细节到用时也会经常感到费解和奇怪。

趁这个时间，把 TypeScript 的语法细节也捋了一下，实践中，也遇到了不少以前忽略的 TypeScript 使用细节。

### 代码和内容组织

编程老司机有个经验：新的技术，仅仅看看文档，跑几个例子，基本上也就能够了解了。不过要想真正了解这个技术的可用性，在学习的基础上，按照自己的想法做点东西，才能够深入的了解这个技术的可用性。

刚开始时，只是想着把这三个技术把Google的例子刷一遍。

## 第一章：概述

这是一个以学习 tensorflow.js 为目的的项目。
使用 React Hooks + Typescript 实现主要界面。

### 技术栈

* TypeScript (v3.7.2)
* React (v16.12) : react-scripts, react-hooks, react-router
* Tensorflow.js (v1.5.2)
* AntD(v4.0.0) & AntV/bizcharts

### 快速开始

    git clone https://github.com/iascchen/react-tfjs-playground.git
    yarn
    yarn start
    
Open [http://loalhost:3000](http://loalhost:3000)

### 目录结构

node    ttjs-node 代码，用于模型训练，比在浏览器中快不少
src     React Tensorflow.js Camp 的 Web App 代码 

public/model    用于存放你自己的模型，可以直接在 Web App 中使用 fetch('/model/...') 获取
public/data     用于存放你自己的数据，可以直接在 Web App 中使用 fetch('/data/...') 获取
public/preload  预先下载好的数据和模型，因为体积太大，并不放在 git 里。打包在 docker 中。
                Web App中使用 fetch('/preload/...') 获取到
                node.js 中使用文件访问相对路径获取，如：'../../public/preload/model'
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

[x] 使用预训练的MobileNet模型 -> 扩展模型 -> 仅训练靠后扩展的几层 -> 新的可用模型

特征的提取 => 人脸特征点

### 数据和模型的保存

[x] 模型存储和上传加载

[x] 数据存储和上传加载

### 交互设计和实现

[x] 图片上传显示组件

[x] 图片分类标注组件

[x] 摄像头组件，拍照上传

[ ] 对视频流的处理

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


