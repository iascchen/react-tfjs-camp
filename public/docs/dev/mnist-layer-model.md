# MNIST 的 LayerModel 实现

## 数据集的获取和加载

### 下载数据集到本地

要完成 MNIST 实验，需要下载对应的数据集。在国内下载速度比较慢（或者需要科学上网），为了减少不必要的等待，我们先将这些数据集下载到本地，以便多次使用。

在命令行中使用以下命令，下载数据。

	$ cd ./public/preload/data
	$ ./download_mnist_data.sh

如果不能执行的话，请检查一下系统是否已经安装 `wget` 。

### 使用 fetch 加载数据文件

加载数据的代码会在多地多次使用，放在 `./src/utils.ts` 中，将 URL 所指示的资源文件，加载到 Buffer 对象中。为了处理 gz 文件，使用了 zlib 包。

	import * as zlib from 'zlib'
	...
	export const fetchResource = async (url: string, isUnzip?: boolean): Promise<Buffer> => {
	    const response = await fetch(url)
	    const buf = await response.arrayBuffer()
	    if (isUnzip) {
	        logger('unzip...', url)
	        return zlib.unzipSync(Buffer.from(buf))
	    } else {
	        return Buffer.from(buf)
	    }
	}

### 数据集的创建

### 图片数据的显示

### 数据集的切换

### 内存的限制

## CNN 网络模型

## 模型训练

### 使用浏览器训练

### 使用 Node.js 训练

### tf-vis的集成

## 推理

### 数字手写板的实现

### 将位图转化成 Tensor





