# 从零开始 Start from scratch

下面的部分内容会逐渐移到 public/docs 目录下。

##环境安装

安装 Node环境

    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash    
    nvm install 13
    node --version

安装 yarn 工具
    
    curl -o- -L https://yarnpkg.com/install.sh | bash
    yarn --version
 
> In China, you can use taobao npm registry

    npm config set registry https://registry.npm.taobao.org
    
## 创建项目    
    
    npx create-react-app react-tfjs-playground --template typescript
    cd react-tfjs-playground
    yarn
    yarn start
    
这是一个经典的 React 启动项目。你可以尝试一下其它的命令：

    yarn test
    yarn build
