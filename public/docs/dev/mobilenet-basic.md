# 站在预训练模型的肩上——以 MobileNet 为例

Google 提供了预训练的 MobileNet 图片分类器，我们可以直接使用它。

## 使用预训练的 MobileNet 图片分类器

参考代码为 `./src/components/mobilenet/MobilenetClassifier.tsx`

### 模型下载

在使用 MobileNet 模型前，需要对预训练模型进行下载。

* 我们可以使用 URL 直接从 Google 下载。

		export const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'

* 也可以提前下载，然后本地加载。

	执行下面的语句下载预训练的模型。下载完成后，可以进入对应的目录，观察一下用于 Tfjs 的模型到底是什么模样。
	
		$ cd ./public/preload/model/
		$ python3 ./tfjs_mobilenet_model_downloader.py

	使用本地 URL 加载：
	
		export const MOBILENET_MODEL_PATH = '/preload/model/mobilenet/mobilenet_v1_0.25_224/model.json'
		
### 用 tf.loadLayersModel 模型加载 

	useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.WAITING)

        let model: tf.LayersModel
        tf.loadLayersModel(MOBILENET_MODEL_PATH).then(
            (mobilenet) => {
                model = mobilenet

                // Warmup the model. This isn't necessary, but makes the first prediction
                // faster. Call `dispose` to release the WebGL memory allocated for the return
                // value of `predict`.
                const temp = model.predict(tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])) as tf.Tensor
                temp.dispose()

                setModel(model)

                const layerOptions: ILayerSelectOption[] = model?.layers.map((l, index) => {
                    return { name: l.name, index }
                })
                setLayersOption(layerOptions)

                setStatus(STATUS.LOADED)
            },
            loggerError
        )

        return () => {
            logger('Model Dispose')
            model?.dispose()
        }
    }, [])

* 使用 `tf.loadLayersModel(MOBILENET_MODEL_PATH)` 加载预训练的 MobileNet 模型及权重。
* 加载后，可以做下模型预热，并非必须，不过可以提升第一次 predict 的速度。

                // Warmup the model. This isn't necessary, but makes the first prediction
                // faster. Call `dispose` to release the WebGL memory allocated for the return
                // value of `predict`.
                const temp = model.predict(tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])) as tf.Tensor
                temp.dispose()
                
* 提取模型的 Layers 信息，用于详细观察 Layers 的具体情况。

                const layerOptions: ILayerSelectOption[] = model?.layers.map((l, index) => {
                    return { name: l.name, index }
                })
                setLayersOption(layerOptions)

## MobileNet 使用的数据集 —— Imagenet

### Imagenet 的 1000 个分类

MobileNet 的训练数据集为 Imagenet，包括 1000 个分类。让我们看看这 1000 个分类是什么，参考代码为 `./src/components/mobilenet/ImagenetClasses.ts`：

	export interface ILabelMap {
	    [index: number]: string
	}

	export const ImagenetClasses: ILabelMap = {
	    0: 'tench, Tinca tinca',
	    1: 'goldfish, Carassius auratus',
	    2: 'great white shark, white shark, man-eater, man-eating shark, ' +
	      'Carcharodon carcharias',
	    3: 'tiger shark, Galeocerdo cuvieri',
	    4: 'hammerhead, hammerhead shark',
	    5: 'electric ray, crampfish, numbfish, torpedo',
	    6: 'stingray',
	    7: 'cock',
	    ...
    }

### 使用 AntD Tags 展示分类

参考代码为 `./src/components/mobilenet/ImagenetTagsWidget.tsx`：

	const ImagenetTagsWidget = (): JSX.Element => {
		...
		return (
			...
                {Object.keys(ImagenetClasses).map((key, index) => {
                    const tag = ImagenetClasses[index]
                    const isLongTag = tag.length > 20
                    const tagElem = (
                        <Tag key={tag}>
                            {isLongTag ? `${tag.slice(0, 20)}...` : tag}
                        </Tag>
                    )
                    return isLongTag ? (
                        <Tooltip title={tag} key={tag}>
                            {tagElem}
                        </Tooltip>
                    ) : (
                        tagElem
                    )
                })}
            ...
		)
	}

### 使用第三方 API 自动翻译

## 推理

### 图片上传显示组件

### 摄像头拍照组件


------------


模型

[x] 使用预训练的MobileNet模型. 获得模型和加载 Weights

[x] 使用预训练的MobileNet模型 -> 特征 -> 机器学习算法 KNN teachable-machine

[x] 使用预训练的MobileNet模型 -> 扩展模型 -> 仅训练靠后扩展的几层 -> 新的可用模型

数据和模型的保存
[x] 模型存储和上传加载

[x] 数据存储和上传加载

交互设计和实现
[x] 图片上传显示组件

[x] 图片分类标注组件

[x] 摄像头组件，拍照上传

[ ] 对视频流的处理