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
	
## React Hooks

下面的内容是使用 React-Scripts 创建一个全新的 React 项目。这些内容记录了如何从零开始，一步一步创建 React-Tfjs-Camp 的主要过程。

React 不多介绍，是由 Facebook 开源，是当前最流行的 Web 前端框架之一。

在这个项目里，使用了 React 16.8 之后推出的新特性 React Hooks。React 团队希望，组件不要变成复杂的容器，最好只是数据流的管道，开发者根据需要，组合管道即可。这种函数化（Function Program）的编程形式，能够大大降低 React 的学习曲线。

关于 React Hooks，已经有了不少中文文章。例如：阮一峰的入门介绍就写得挺好。在后续的内容中，对于一些初级使用，我不会做太多展开，重点会记录在 React-Tfjs-Camp 的实现过程中，遇到的一些典型问题，以及是如何使用合适的方式进行解决的。

参考链接：

* React-Scripts  [https://create-react-app.dev/](https://create-react-app.dev/)
* React Hooks 官方链接 [https://reactjs.org/docs/hooks-intro.html](https://reactjs.org/docs/hooks-intro.html)
* 阮一峰的入门介绍 [React Hooks 入门教程](https://www.ruanyifeng.com/blog/2019/09/react-hooks.html)。

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

### 项目目录简述

使用 `ls -la` 能够看到项目中生成了这些文件。 

	$ ls -la
	total 936
	drwxr-xr-x    11 chenhao  staff     352  4 10 16:13 .
	drwxr-xr-x    14 chenhao  staff     448  4 10 16:12 ..
	drwxr-xr-x    12 chenhao  staff     384  4 10 16:13 .git
	-rw-r--r--     1 chenhao  staff     310  4 10 16:13 .gitignore
	-rw-r--r--     1 chenhao  staff    2097  4 10 16:13 README.md
	drwxr-xr-x  1036 chenhao  staff   33152  4 10 16:14 node_modules
	-rw-r--r--     1 chenhao  staff     904  4 10 16:13 package.json
	drwxr-xr-x     8 chenhao  staff     256  4 10 16:13 public
	drwxr-xr-x    11 chenhao  staff     352  4 10 16:13 src
	-rw-r--r--     1 chenhao  staff     491  4 10 16:13 tsconfig.json
	-rw-r--r--     1 chenhao  staff  461880  4 10 16:13 yarn.lock
	
这些文件的简要说明如下：
	
* .git——使用 Git 的配置信息目录。
* .gitignore——那些文件不需要被 Git 进行版本管理。
* package.json——Node.js 项目中最重要的文件，关于依赖包、运行脚本，统统放在这里。
* tsconfig.json——使用 TypeScript 所需的配置文件，用于设置 TypeScript 支持的语法特性。
* public——存放 Web APP 中的静态页面和资源文件。
* src——代码目录。
* node_modules——下载和安装的 npm 包。运行 `yarn` 时检查 package.json 的信息来安装。
* yarn.lock——使用 `yarn` 安装生成的 npm 包依赖文件。当项目中发生依赖包冲突的时候，可以通过修改和调整它来解决。