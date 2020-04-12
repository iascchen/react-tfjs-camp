# 从零开始 Start from Scratch

以下操作均在 MAC OS 上验证。

参考文档尽量都看官方最新的。其他地方的文档和教程，往往会随着时间的流逝而陈旧过时，本文亦如是。

## 环境安装

### 安装 Node 环境

使用 nvm 便于管理 node.js 的版本更替。参考链接 [https://github.com/nvm-sh/nvm](https://github.com/nvm-sh/nvm)。

写此文档时 Node.js 最新的版本是 13.12。参考链接 [https://nodejs.org/](https://nodejs.org/)

在命令行中输入以下命令。其中 `$` 为命令行提示符，请不要输入。

	$ curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash    
	
	$ nvm install 13
	
	$ nvm list
	->     v13.12.0
	default -> v13.12.0
	node -> stable (-> v13.12.0) (default)
	stable -> 13.12 (-> v13.12.0) (default)
	iojs -> N/A (default)
	
	$ node --version
	v13.12.0

### 安装 yarn 工具

yarn 是一个很方便的包管理工具，可以替代 npm。如果您需要快速了解 npm ，可以参考链接 [安装Node.js和npm](https://www.liaoxuefeng.com/wiki/1022910821149312/1023025597810528)

yarn 会缓存每个下载过的包，所以再次使用时无需重复下载。同时利用并行下载以最大化资源利用率，因此安装速度更快。参考链接 [https://classic.yarnpkg.com/en/docs/](https://classic.yarnpkg.com/en/docs/)
    
	$ curl -o- -L https://yarnpkg.com/install.sh | bash
	$ yarn --version
	1.22.4
	
常用 yarn 命令有：

* `yarn` 安装 package.json 中的包，相当于 `npm install`。
* `yarn add [package]` 安装指定的 npm 包, 相当于 `npm install [package]`。
* `yarn start` 运行 package.json 中的命令脚本, 相当于 `npm run start`。 

> 在国内，直接从源头安装 npm 包有时会比较慢，可以使用淘宝的 npm 镜像加速。使用下面的命令设置和查看

	$ npm config set registry https://registry.npm.taobao.org
	
	$ npm config list
	; cli configs
	metrics-registry = "https://registry.npm.taobao.org/"
	scope = ""
	user-agent = "npm/6.14.4 node/v13.12.0 darwin x64"
	; userconfig /Users/chenhao/.npmrc
	registry = "https://registry.npm.taobao.org/"
	...
	
## React 和 React Hooks

使用组件构建 Web APP，是当前前端开发的最佳实践之一。从本质上说，就是将你的应用分拆成一个个功能明确的模块，每个模块之间可以通过合适的方式互相组合和联系，形成复杂的前端 Web 应用。

比较流行的组件框架有 Facebook 开源的 React，还有一个是国人尤雨溪开源的 Vue。想了解这两个框架的基本差异的同学，可以阅读一下知乎上的 [React VS Vue：谁会成为2020年的冠军](https://zhuanlan.zhihu.com/p/89416436)

自从 React 诞生后，其创建组件的方式从 ES5 时期声明式的 createClass ，到支持原生 ES6 class 的 OOP 语法，再到发展出 HOC 或 render props 的函数式写法，官方和社区一直在探索更方便合理的 React 组件化之路。随之而来的一些问题是：

* 组件往往变得嵌套过多
* 各种写法的组件随着逻辑的增长，变得难以理解
* 尤其是基于类写法的组件中，this 关键字暧昧模糊，人和机器读起来都比较懵，难以在不同的组件直接复用基于 state 的逻辑
* 人们不满足于只用函数式组件做简单的展示组件，也想把 state 和生命周期等引入其中

Hooks 是 React 16.8 之后推出的新特性，React 团队希望，组件不要变成复杂的容器，最好只是数据流的管道，开发者根据需要，组合管道即可。
这种函数化（Function Program）的编程形式，能够大大降低 React 的学习曲线。
属实讲，挺香的。

关于 React Hooks，已经有了不少中文文章。例如：阮一峰的入门介绍就写得挺好。在后续的内容中，对于一些初级使用，我不会做太多展开，重点会记录在 React-Tfjs-Camp 的实现过程中，遇到的一些典型问题，以及是如何使用合适的方式进行解决的。

参考链接：

* React Hooks 官方链接 [https://reactjs.org/docs/hooks-intro.html](https://reactjs.org/docs/hooks-intro.html)
* 阮一峰的入门介绍 [React Hooks 入门教程](https://www.ruanyifeng.com/blog/2019/09/react-hooks.html)。
* 深入剖析可以读一下 [React Hooks 深入不浅出](https://segmentfault.com/a/1190000017182184)

### 创建 React 应用

下面的内容是使用 React-Scripts 创建一个全新的 React 项目。这些内容记录了如何从零开始，一步一步创建 React-Tfjs-Camp 的主要过程。

参考链接：React-Scripts  [https://create-react-app.dev/](https://create-react-app.dev/)

### 创建一个新的 React 项目

`yarn create react-app` 用于创建 React App，等于 npm 原生命令的 `npx create-react-app`。

`--template typescript` 的参数，表明使用 typescript 作为编程语言。

	$ yarn create react-app react-tfjs-new --template typescript
	$ cd react-tfjs-new
	
执行 `yarn start` 之后，就能够通过 [http://localhost:3000](http://localhost:3000) 访问这个新的项目了。
	 
	$ yarn start
	Compiled successfully!
	You can now view react-tfjs-new in the browser.
	Local:            http://localhost:3000
	   
你还可以尝试一下其它的命令。

	$ yarn test
	$ yarn build

### React 项目目录简述

使用 `ls -la` 能够看到项目中生成一些文件。

	.
	|____.git					使用 Git 的配置信息目录
	|____.gitignore			哪些文件不需要被 Git 进行版本管理
	|____README.md
	|____node_modules			下载和安装的 npm 包。运行 `yarn` 时检查package.json 的信息来安装 
	|____yarn.lock			使用 `yarn` 安装生成的 npm 包依赖文件。当项目中发生依赖包冲突的时候，可以通过修改和调整它来解决
	|____package.json			Node.js 项目中最重要的文件，关于依赖包、运行脚本，统统放在这里
	|____tsconfig.json		使用 TypeScript 所需的配置文件，用于设置 TypeScript 支持的语法特性
	|____public				存放 Web APP 中的静态页面和资源文件
	|____src					Web APP 源代码
