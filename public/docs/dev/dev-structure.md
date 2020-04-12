# 构建 React 开发框架

在上一篇中，我们使用 React-Scripts 创建了一个新的 React APP。现在开始装修改造。

## React-tfjs-camp 的目录结构

React-tfjs-camp 对目录结构作了如下调整：

	.
	|____.eslintrc.js		使用 eslint 进行代码格式检查的配置文件 
	|____.git
	|____.gitignore
	|____README.md
	|____LICENSE			此项目使用 MIT LICENSE 的说明
	|____node_modules
	|____yarn.lock
	|____package.json
	|____tsconfig.json
	|____public				静态资源目录
	|____src				Web APP 代码目录
	|____node				Node.js 代码目录
	|____Dockerfile			用于构建 Docker Image
	|____docker_build.sh	Docker 构建脚本 
	|____docker_run.sh		Docker 运行脚本

### public 目录结构

	.
	|____favicon.ico
	|____index.html
	|____404.html
	|____manifest.json
	|____robots.txt
	|____images				根目录的静态图片目录
	|____docs				文档目录
	| |____images			文档所使用的图片
	| |____ai				AI Concept 相关文档
	| |____dev				Develop Tutorial 相关文档
	|____model				用户通过 Web App 或者 Node.js 训练后的模型，可以拷贝到这里，供下次使用
	|____data				用户代码中使用的个性化数据，可以放置在此处，供下次加载使用
	|____preload			项目使用的各种数据集和预训练模型文件，下载一次之后，减少不必要的网络延迟
	| |____model
	| | |____download_model.sh		下载模型的脚本
	| |____data
	| | |____download_data.sh		下载数据集的脚本

### src 目录结构

	.
	|____index.tsx				React 入口，整个项目的 Web 渲染由此开始
	|____index.css		
	|____App.tsx				App Root 组件
	|____App.css				App 组件相关格式 css
	|____serviceWorker.ts		Service Worker 相关，未修改
	|____routers.ts				使用 React-Router 集中处理页面路由
	|____constant.ts			一些常量
	|____utils.ts				一些工具常数和函数
	|____App.test.tsx			APP Unit Test 入口，未修改
	|____setupTests.ts			未修改
	|____components				主要代码在这里，定义了 Web APP 所用到的页面组件
	| |____common
	| | |____visulization
	| | |____tensor
	| | |____tfvis
	| |____curve
	| |____iris
	| |____mnist
	| |____mobilenet
	| |____rnn
	| |____pretrained
	|____react-app-env.d.ts		一些没有声明类型的 npm 包，需要放在这里，才可以被 Typescript 正确编译
	|____typescript_logo.svg
	|____react_logo.svg
	
### node 目录结构
	.
	|____README.md
	|____package.json			使用 Node.js 代码的 Package.json
	|____tsconfig.json			使用 Node.js 代码的 Typescript 语法特性设置
	|____node_modules
	|____src					Node.js 代码目录
	| |____jena
	| |____sentiment
	| |____simpleObjDetector
	| |____textGenLstm
	| |____utils.ts				一些工具常数和函数
	|____logs					用于存放训练产生的 logs


## 规范代码语法和风格检查

这部分内容非常重要，不过却往往被各种开发教程忽略，可以让我们避免使用在 JS 中广受诟病的那些陈旧语法和奇技淫巧，提高代码的可读性，减少代码的漏洞。

### tsconfig.json

使用 tsconfig.json 对 Typescript 语法特性设置。这个文件会在使用 `tsc` 进行  TypeScript 语法检查和编译时起作用。

在 React-tfjs-camp 的 Web APP 中，使用了 React 和 ES6 语法特性，设置如下。其中 `...` 略去的部分主要对代码格式进行规范化限制的部分。简单来说，这些设置使得我们能够使用诸如：import/export、箭头函数、async/await 等较新的 JS 语法。

	{
	    "compilerOptions": {
	        "allowJs": false,
	        "module": "esnext",
	        "jsx": "react",
	        "target": "es6",
	        "lib": [
	            "dom",
	            "dom.iterable",
	            "es6",
	            "es7",
	            "esnext"
	        ],
			...
	    },
	    "include": [
	        "src"
	    ],
	    "exclude": [
	        "node_modules"
	    ]


针对于 Node.js 的代码，使用的配置有如下修改：

	{
	    "compilerOptions": {
	        "allowJs": true,
	        "module": "commonjs",
	        "target": "es6",
	        ...
	}
	
### .eslintrc.js

.eslintrc.js 是 eslint 的配置文件，被用于进行代码风格检查，在开发的 IDE 中使用。下面的设置，集成了常用的 Typescript、React 推荐代码风格检查规则。

在一些文档中，你还会看到使用 tslint 进行 Typescript 的代码检查。当前，Typescript 官方已经推荐使用的是 eslint。

	module.exports = {
	    root: true,
	    parser: '@typescript-eslint/parser',
	    plugins: [
	        '@typescript-eslint', "react", 'react-hooks', 'eslint-comments'
	    ],
	    extends: [
	        "react-app",
	        'eslint:recommended',
	        'plugin:@typescript-eslint/eslint-recommended',
	        'plugin:@typescript-eslint/recommended',
	        "plugin:react/recommended",
	        'standard-with-typescript',
	    ],
	    parserOptions: {
	        project: "./tsconfig.json",
	        sourceType:  'module',  // Allows for the use of imports
	    },
	    rules: {
	        "react-hooks/rules-of-hooks": "error",
	        "react-hooks/exhaustive-deps": "warn",
	        "@typescript-eslint/interface-name-prefix": ["error", {"prefixWithI": "always"}],
	        "@typescript-eslint/indent": ["error", 4, { 'SwitchCase': 1 }],
	        "jsx-quotes": ["error", "prefer-single"],
	        '@typescript-eslint/no-unused-vars': ['error', {
	            'vars': 'all',
	            'args': 'none',
	            'ignoreRestSiblings': true,
	        }],
	        "@typescript-eslint/strict-boolean-expressions": 0,
	    },
	    settings:  {
	        react:  {
	            version:  'detect',  // Tells eslint-plugin-react to automatically detect the version of React to use
	        },
	    }
	};
	
## 改造页面布局

`/src/App.tsx` 是 React App 中常用的根页面组件。

### React 函数化组件

我们先来看一下 App.tsx 页面的结构，这是一个最简单的 React 函数组件的例子。

	import React, { ... } from 'react'
	
	const App = (): JSX.Element => {
        ...	
	    return (
	        <Layout>
	            ...
	        </Layout>
	    )
	}
	
	export default App

* `import React, { ... } from 'react'` 语句声明了当前组件所依赖 React 包。
* `const App = (): JSX.Element => { ... }` 声明了这是一个名为 `App` 的页面函数组件，这个组件的返回值是 JSX.Element 类型。
* `return (<Layout>...</Layout>)` 这段代码展示的是具体返回的 JSX.Element 由哪些页面组件和元素组成。
* `export default App` export 输出的内容，才能够被其他组件引用。非 default 的输出，需要在 import 时放在 `{}` 中。

### 使用 Ant Design 构建页面框架

Ant Design 是蚂蚁金服体验技术部推出的一个服务于企业级产品的设计体系。如果您对于交互界面没有特殊的视觉效果设计要求，使用 AntD 是个不错的选择。在实际应用中，AntD 常常被用于 Web 应用的管理后台开发，能够非常快的搞定交互界面。

Ant Design v4 于 2020 年的 2 月 28 日正式发布，当前(码字时)版本已经升级到 v4.1.2 了。Ant Design 4 有较大的提升，最重要的更新在于全面支持 React Hooks，重写了 Form、Table 等关键组件的实现，使用起来代码优美了不少。

AntD 的文档非常易于理解和使用。参考链接 [https://ant.design/index-cn](https://ant.design/index-cn)

参照 AntD 的官方文档，很快就能够搭出 React-tfjs-camp 的页面框架。

#### 在项目中使用 AntD

在项目根目录中，执行以下命令，安装 AntD 包。安装完成之后，package.json 中会自动增加 `"antd": "^4.1.2",` 的依赖包条目。

	$ yarn add antd

打开 `/src/index.tsx`, 在文件头部的声明部分增加 antd.css，以使用 AntD 定义的页面风格资源。

	import 'antd/dist/antd.css'
	
在需要的页面 import 所需使用 AntD 组件即可。

	import { ... } from 'antd'

###  页面布局

使用 AntD 的 Layout 组件，能够帮助我们非常容易的构建出各种结构的应用框架。React-tfjs-camp 采用左右结构的页面布局，如下图所示。

![页面布局](../images/dev_layout_ui.png)

左侧的菜单条被封装在 SideBar 里，页面主体被封装在 BodyContainer 里。修改 `/src/App.tsx` 如下：

	import React, { useState } from 'react'
	import { Layout } from 'antd'
	
	import './App.css'
	import SideBar from './components/common/SideBar'
	import BodyContainer from './components/common/BodyContainer'
	...
	const { Content, Header, Sider, Footer } = Layout
	
	const App = (): JSX.Element => {
        ...	
	    return (
	        <Layout>
	            ...
	                <Sider collapsible collapsed={sCollapsed} onCollapse={onCollapse}>
	                    <SideBar/>
	                </Sider>
	                <Layout className='site-layout'>
	                    <Header style={{ background: '#fff', padding: '0' }}>
	                        ...
	                    </Header>
	                    <Content style={{ margin: '16px' }}>
	                        <BodyContainer/>
	                    </Content>
	                    <Footer style={{ textAlign: 'center' }}>©2020 Created by Iasc CHEN(iascchen@gmail.com)</Footer>
	                </Layout>
	            ...
	        </Layout>
	    )
	}
	...

## 边栏菜单导航

### AntD Layout Sider

边栏菜单的实现使用了 AntD Layout 中的 Sider 组件。

	<Sider collapsible collapsed={sCollapsed} onCollapse={onCollapse}>
		<SideBar/>
	</Sider>

 * `collapsible` 属性说明了它可以折叠与展开
 * `collapsed` 指示折叠状态，它的值被设定为 sCollapsed
 * `onCollapse` 函数是对应折叠按钮点击的响应方法

### 使用 React Hooks 的 useState 管理边栏状态

App.tsx 需要保存 Sider 组件的折叠状态。这里用到了 Hooks 的 useState。

	import React, { useState } from 'react'
		...
	const [sCollapsed, setCollapsed] = useState(true)
		...
	const onCollapse = (): void => {
		setCollapsed(collapsed => !collapsed)
	}

* `const [sCollapsed, setCollapsed] = useState(true)` 声明了一个名为 `sCollapsed` 的状态变量，对其进行赋值的函数为 `setCollapsed`，这个状态的初始值为 `true`
* `setCollapsed`的参数，可以是具体的一个值，也可以是一个回调函数。如果新的 state 需要通过使用先前的 state 计算得出，那么可以将回调函数当做参数传递给 setState。该回调函数将接收先前的 state，并返回一个更新后的值。
* 个人的 Tips：将所有的 State 变量，以 `s` 开头命名，在引用的时候便于和局部变量区分。

**请注意：**

useState 和后面介绍的其他的 React Hooks 声明一样，都需要放在组件函数的**前部**，才能被正确使用，这是由 Hooks 使用队列实现的原理决定的。更多使用 Hooks 的规则细节请参考[Invalid Hook Call Warning](https://reactjs.org/warnings/invalid-hook-call-warning.html)。

### 用 React-Route 实现页面路由跳转

使用 React 构建的单页面应用，要实现页面间的跳转，需要使用页面路由切换——React-Route。

在项目中增加 React-Route 相关的包，后两个 @types 包是 TypeScript 的需要：

	$ yarn add react-router-config react-router-dom @types/react-router-config @types/react-router-dom

`/src/App.tsx` 使用 BrowserRouter 将需要进行路由的页面部分包起来。

	import React, { useState } from 'react'
	import { BrowserRouter as Router } from 'react-router-dom'
	...
	
	const App = (): JSX.Element => {
	    ...
	    return (
	        <Layout>
	            <Router>
	                <Sider collapsible collapsed={sCollapsed} onCollapse={onCollapse}>
	                    <SideBar/>
	                </Sider>
	                <Layout className='site-layout'>
	                    ...
	                        <BodyContainer/>
	                    ...
	                </Layout>
	            </Router>
	        </Layout>
	    )
	}
	...

`/src/components/common/SideBar.tsx` 使用了 AntD 的 Menu 组件设置边栏菜单格式，用 react-route-dom 的 Link 设置页面之间的路由关系。

	import React from 'react'
	import { Link } from 'react-router-dom'
	import { Menu } from 'antd'
	...
	
	const { Item, SubMenu } = Menu
	
	const SideBar = (): JSX.Element => {
	    return (
	        <div>
	            <header className='App-header'>
	                <Link to='/'>
	                    <img src={logo} className='App-logo' alt='logo'/><h2 style={{ color: 'white' }}>RTCamp</h2>
	                </Link>
	            </header>
	            <Menu theme='dark' mode='inline'}>
	                <SubMenu title={<span><LineChartOutlined/><span>逻辑回归 Logisttc </span></span>}>
	                    <Item key='1.1'>
	                        <Link to='/curve'><span> 曲线拟合 Curve </span></Link>
	                    </Item>
	                    ...
	                </SubMenu>
	                ...
	            </Menu>
	        </div>
	    )
	}
	...

`/src/components/common/BodyContainer.tsx` 使用 react-router-config 包里的 renderRoutes，我们可以将集中设置在 routers.ts 中的路由映射，对应到页面框架里的 BodyContainer 里。

	import React from 'react'
	import { renderRoutes } from 'react-router-config'
	import { Alert } from 'antd'
	
	import routes from '../../routers'
	
	const { ErrorBoundary } = Alert
	
	const BodyContainer = (): JSX.Element => {
	    return (
	        <div style={{ padding: 24, background: '#ffffff', minHeight: '80vh' }}>
	            <ErrorBoundary>
	                {renderRoutes(routes)}
	            </ErrorBoundary>
	        </div>
	    )
	}
	
	export default BodyContainer


所有的路由映射都被定义在 `/src/routers.ts` 中。这么做的好处是便于维护管理，让组件可以专注于自己的功能逻辑。

	import { RouteConfig } from 'react-router-config'
	
	import Home from './components/common/Home'
	import Curve from './components/curve/Curve'
	...
	
	const routes: RouteConfig[] = [
	    { path: '/', exact: true, component: Home },
	    { path: '/curve', component: Curve },
	    ...
	
	    { path: '*', component: Home }
	]

* 设置 `'/'` 映射时，使用 `exact: true` 以表明不会“误杀”其他以 `'/'` 开头的路由设置。你可以试试 `exact: false` ，或者把这个设置去掉，看看会出现什么结果。
* 设置 `'*'` 映射，对无效的页面路由统一处理，都映射到 Home。试试去掉这个映射，看看会出现什么😄
	
我们只用到了 React-Route 的一点点基础部分。关于 React-Route 的更多内容可以参考：

* [官方Github](https://github.com/ReactTraining/react-router)
* [React-Router 的 Hooks 实现](https://blog.csdn.net/weixin_43870742/article/details/102966040)

## ErrorBoundary

你有没有注意到，在 `/src/components/common/BodyContainer.tsx` 中，我们为 Content 封装一个 ErrorBoundary，用于截获在 Web APP 运行时，没能被 Catch 到的异常和错误，对它们进行统一的显示。

目前，React 官方还没有实现 getDerivedStateFromError、componentDidCatch 这些用于错误和异常处理的函数，所以只能够采用 React 类组件来完成这个功能。参考文档 [错误边界](https://zh-hans.reactjs.org/docs/error-boundaries.html)

AntD 对 React 官方文档中的 ErrorBoundary 做了封装，我们可以直接使用。

**请注意** 在开发模式下，ErrorBoundary 显不出效果。



