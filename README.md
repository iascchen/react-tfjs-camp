# react-tfjs-camp 全栈 AI 训练

这是一个以学习 tensorflow.js 为目的的项目。使用 React Hooks + Typescript 实现主要界面。

BTW，这是个平民 AI 教程，可以不使用 GPU 完成。

## 快速开始

### 技术栈

* TypeScript (v3.7.2)
* React (v16.12) : react-scripts, react-hooks, react-router
* Tensorflow.js (v1.5.2) 才一个多月的功夫，已经升级到 v1.7.0 了
* AntD(v4.0.0) & AntV/bizcharts

### 快速开始

* Web APP

	    git clone https://github.com/iascchen/react-tfjs-camp.git
	    yarn
	    yarn start
    
	Open [http://loalhost:3000](http://loalhost:3000)

* 使用 Node.js 的代码部分

		cd node
		yarn
		ts-node ./src/**.ts

* 使用 Docker （还没上传呢）

	TODO
	
### 目录结构

	node    tfjs-node 代码，用于部分模型训练，比在浏览器中快不少。
	src     React Tensorflow.js Camp 的 Web App 代码 
	
	public/model    用于存放你自己的模型，可以直接在 Web App 中使用 fetch('/model/...') 获取
	public/data     用于存放你自己的数据，可以直接在 Web App 中使用 fetch('/data/...') 获取
	public/preload  预先下载好的数据和模型，因为体积太大，并不放在 git 里。打包在 docker 中。
	                Web App中使用 fetch('/preload/...') 获取到
	                node.js 中使用文件访问相对路径获取，如：'../../public/preload/model'

## 缘起

多灾多难的 2020 年，必定会在历史上记一笔。春节以来，COVID-19 疫情日益严重，全国人民都在家隔离。憋在家里的这段日子，每天被各种信息坏消息和好消息蹂躏来，蹂躏去。
时间久了，也渐渐应激疲劳，静下来，找点事，排遣一下时间。

最近几年，我一直在关注如何让用户拥有通过私有数据训练，实现 “私有AI” 能力的方法。不过没有找到特别理想的方案，直到看到 Tensorflow.js 的最近更新。

我们先来看看当前经典的 AI 使能架构：

[前端AI应用+模型Serving+后端计算的示意图]

你的所有数据都会被传送到后端服务器，再使用服务端资源进行训练。在大数据和 AI 面前，你完全没有隐私可言，隐私换智（fang）能（bian），就是当前的现状。

另外，对于小型的 AI 创新服务企业而言，这种模式需要更多的资源才能实施，一是数据传输量大，二是服务端计算资源要求也不老少。

相对理想的架构是将一些的 AI 计算放在客户端，也就是大家常说的“边缘计算”。在这种模式下，客户端在本地上具有足够的“智能”，除了能够进行必要的事件处理，过滤“无用(也许吧)”的数据，更妙的是在客户端可以进行“小数据”个性化训练，实现“私有 AI”。在这种架构下，服务端负责提供在“大数据”上预先训练好的基础模型，客户端进行“小数据”上的私人订制，既可以保持隐私，也能够享受到更智能私人服务。

### Tensorflow.js

Tensorflow.js 在 2月4日 除夕那天发布了 v1.5.2，能够在浏览器端做训练了！！！能直接在浏览器端做训练，即使只是“简单”的训练，是个很有意义的大进步。可以在用户（浏览器）端进行个性化的训练，进行隐私信息的脱敏，再和服务端更强大的AI能力相配合，相信不久之后，我们能够看到更多既能保持个人数据隐私，也足够智能灵活的新型AI应用。Google也提供了“联邦计算”的例子，也许这是“私人AI”时代的曙光。

[后端预训练+前端个性化训练+联邦计算的示意图]

除此之外，模型加载、存储和转换相关功能基本完备，能够很顺畅地与 Tensorflow v2.0 以上版本训练所产生的模型进行对接。

### React Hooks

因为工作性质的关系，我自己的主要时间都放在了团队管理、技术预研和架构设计方面，虽然也动手做一部分代码，但是对于不少新的技术特性细节，都得靠团队小伙伴们的鼎力支持了。

我的团队主要使用基于 JavaScript/TypeScript 的 React + Redux + Express 进行全栈开发，除了人工智能后端用 Python。

使用 React 已经是相当舒适的编程体验了，不过如果涉及到复杂的组件间信息交互，以及多层父子之间的数据共享，就必须使用高阶组件+Context、或者 Redux Connect 等形式传递数据，这样的代码风格，并不简洁和优美。

React Hooks 是 React 在 16.8 之后支持的新的特性，发布至今已经一年了吧。因为当前工作中的项目里一直使用的是 Component 的对象形式，一直没有太仔细的学习和使用。React Hooks 更容易将组件的 UI 与状态分离，不用再写很看起来不爽的 setState，不用担心在 componentDidMount、componentWillUnmount 忘了做这做那。使用 React Hooks 开发，可以用很直观的数据变化驱动界面变化的思路来组织代码，实际使用，还真香。

### TypeScript

工作中，只要求团队在后端开发使用 TypeScript。

作为一个资深老 Java，最近几年，使用 JavaScript 和 Python 居多，确实有一种放飞自我的感觉。源自于 Java 开发养成的代码习惯，加上 JS 和 Python 的灵活方便，大大提高了我将创意和想法快速开发实现的效率。不过，因为经常在多种语言之间切来切去，不少语法细节到用时也会经常感到费解和奇怪。

趁这个时间，把 TypeScript 的语法细节也捋了一下，实践中，也遇到了不少以前忽略的 TypeScript 使用细节。

## 第一季 基础

学习新的技术，仅仅看看文档，跑几个例子，基本上也就能够了解了。不过要想真正深入了解这个技术的优缺点，在初步学习的基础上，还得按照自己的想法做点东西。

这个项目的开始比较随性，列了个 AI 技术实践点的列表，起了个Repo，只是想着把 TFJS Example 的典型例子重刷一遍，能够留下些学习的记录，希望对大家有些帮助。

这个过程，以前用 Python 曾经撸过一遍，学习后体会是各个例子比较散。仅仅是这样去学习 TF 与 AI，往往是关注模型，多过关注问题本身。算法和模型天天都在进化，日日都有新论文发出来，要理解 AI 相关的技术，还是需要思考一些更本质的东西（当然，我的理解也不全面）。

在做了几个例子之后，我开始对这个程序的内容和结构做调整。一年多前，曾经写过一张 PPT，想为 AI 的初学者们，提供一个“端到端”的学习和体验的平台。所谓端到端，就是从领域问题出发，思考所需的数据，运用相应的模型，执行训练，验证推理，形成新应用。这个过程，我自己理解是“从数据出发”的科学探究方法的延伸，就像是古人观测天象、记录物理现象一样。而未来和过去不一样的地方在于，我们可以利用计算机和 AI ，处理更大量的数据，发现隐藏的更深的规律。

[端到端 AI 的PPT截图]

基于这个考虑，对于选取记录的例子，也重新做了一些甄选，初步形成了当前的框架。

* 第一部分，AI 来做加减乘除。从传统的实验数据记录处理问题出发，以曲线拟合、经典的 IRIS 鸢尾花为例，介绍了 Tensor、神经元模型、线性回归、以及多层感知机。
* 第二部分，AI 识数。用 AI 学习的 Hello World 程序 MNIST，介绍了这个例子的数据加载、体验使用不同模型计算的差别，还提供了一个手写数字识别画板，可以直接看到学习前后，模型给出结果的差异。对于 MNIST 的训练学习部分，利用 Tensorflow 的 高级模型 Keras 和基础模型 Graph 分别做了实现。这里面有个满是眼泪的背景：MNIST 层次模型的训练，在浏览器中竟然不收敛（可能是某些参数还需要调整，待我慢慢试来）。所以，中间还会穿插直接使用 Node.js(其实是 ts-node)进行训练的一部分内容。 
* 第三部分，迁移学习。以 Mobilenet 模型为基础，重点讨论的是如何使用预训练的模型，介绍了四个例子。分别是：直接使用 Mobilenet 执行图片分类、Mobilenet + 机器学习 KNN 实现 Teachable Machine、在 Mobilenet 的基础上进行模型扩展实现个性化的图片分类器、以及扩展 Mobilenet以进行简单的对象识别的例子。之所以选择这些例子，一方面是在家笔记本里没有 GPU；另一方面，这些例子更多的体现了边缘计算 AI 应用的特点，在后台大数据计算形成的基础模型之上，利用用户端的计算能力，完成个性化的新任务。
* 第四部分，处理连续数据。语言、音乐、文字、侦测信号，都是连续数据的典型应用。使用 RNN 进行气象预报、语义分析、以及文本生成。

这四个部分，是结合 Tensorflow.js 学习 AI 的最基础内容，所有的内容，都包括如何使用 JS 完成相应的界面的交互。希望这个 Repo 能够成为 AI 学习的练习场，在实现过程中，也会穿插所需要的前后端技术，能够为想了解全栈开发技术的同学提供帮助，也可以形成一个 AI 工具集。

当前例子主要来自于 Google 的 tfjs-examples，因为整体结构不同，以及使用的技术栈不同，做了改写。以后会增补些具有中国特色的例子。

参考书目：

* 《Deep Learning with Python》 Manning
* 《Deep Learning with JavaScript》 Manning
* 《Deep Learning》 花书
* 逐渐补充

具体的内容会放在 public/docs 目录下。这些内容，在运行程序后，会展示在“问题”标签页面下。
 
> 20200312: 应该是从 20200215 正式确定写这个 Repo。眨眼就一个月了，每天吃吃睡睡、宅家锻炼、处理些公司里的事、做做实验，与龟速的机器和越来越高的墙作斗争。接下来应该还能有半个月的空闲时间，先把第一季各部分的教程写完，并重新优化代码。

### 从零开始 Start from scratch

下面的部分内容会逐渐移到 public/docs 目录下。

#### 环境安装

安装 Node环境

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash    
    nvm install 13
    node --version

安装 yarn 工具
    
    curl -o- -L https://yarnpkg.com/install.sh | bash
    yarn --version
 
> In China, you can use taobao npm registry

    npm config set registry https://registry.npm.taobao.org
    
#### 创建项目    
    
    npx create-react-app react-tfjs-playground --template typescript
    cd react-tfjs-playground
    yarn
    yarn start
    
这是一个经典的 React 启动项目。你可以尝试一下其它的命令：

    yarn test
    yarn build
    
#### 构建程序框架

[x] React 概述

[x] 简介 React-Hooks， React-Route

[x] 代码风格检查 ESLint，tsconfig.json。

[x] 使用 AntD 构造了一个左右结构的程序框架

[x] MD 文档显示Widget
    
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

## 第二季 TODO

先立 Flag 吧。如果有时间做第二季，希望包括的内容如下。主要以应用实例为主，不会拘泥于在浏览器端训练，不过估计没有 GPU 很难玩转了。

还会考虑增加利用图形化拖拽构建模型的实现。或者增加一些 3D 可视化的东西（例如集成 tensorspace ）

### GAN 生成对抗网络

### 对话机器人

### 强化学习：玩游戏

metacar-project.com

### 遗传算法赛车：玩游戏

### 博弈：MiniGO

https://github.com/tensorflow/minigo 

### 联邦学习



