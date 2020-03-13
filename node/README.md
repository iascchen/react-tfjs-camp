# 执行 tf-node 程序 

## 

## 执行

    cd node
    yarn
    ts-node ./src/**/*.ts

说明：`ts-node` 会找里当前运行目录最近的 `tsconfig.json` 文件。因此，如果此项目的根目录下运行这些程序，会因为使用的 tsconfig 不对而报错。
