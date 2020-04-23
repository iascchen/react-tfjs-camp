# 初步了解 Tensorflow.js

TensorFlow.js 是 Goolge Tensorflow 的 JS 版本，将高性能机器学习能力带到 JS 世界。

通过《曲线拟合》这个例子，我们对 Tensorflow.js 进行初步了解。在这个例子中，我们要构建一个一元二次方程，并观察使用人工神经网络模型求解的过程。

![../images/curve.mp4](../images/curve.mp4)

## 使用 Tensorflow.js 的几点须知

官方文档 [平台和环境](https://www.tensorflow.org/js/guide/platform_environment)
中，描述了使用 tfjs 的须知，下面的内容列举了其中必须了解的几点。在此，我们先做介绍，稍后在代码中体现。
 
### Backend —— 为什么我的 tfjs 运行很慢？

慢这件事，到底什么原因造成的，确实不好讲。不过了解 Tensorflow.js 运行 backend 的一些背景，会有些帮助。

TensorFlow.js 开发的程序运行时，所有的配置被统称为环境。它包含一个全局的backend，以及一些可以精确控制 TensorFlow.js 特性的标记。

TensorFlow.js 有两种工作平台：浏览器和 Node.js。不同平台有很多不同的配置，平台间的差异影响着基于平台的应用开发。

* 在浏览器平台上，TensorFlow.js 既支持移动设备，也支持台式设备，使用 WebGL API 作为 backend，自动检测并做相应的优化配置。你可以检查一下，浏览器中是否已开启“硬件加速模式”。

![在Chrome中打开硬件加速](../images/dev-tfjs-webgl.png)

* 在 Node.js 平台上，TensorFlow.js 支持直接使用 TensorFlow 的 C 语言 API 来加速操作，它会尽可能使用机器的 GPU 硬件加速模块，如 CUDA。也支持更慢的 CPU 环境。

### 内存管理 —— 这样避免我的程序内存溢出？

使用 WebGL backend 时，**需要显式管理内存**。因为存储Tensor的WebGL纹理，不会被浏览器的垃圾收集机制自动清理。

* 调用dispose()清理tf.Tensor占用的内存
* 在应用中，经常需要把多个操作组合起来。TensorFlow.js提供tf.tidy()方法，可以将  多个操作组合封装在函数中。函数返回时，会清理不再需要的tf.Tensor，这就好像函数执行后，本地变量都会被清理一样。

在一些非 WebGL 环境，有自动垃圾回收机制，在这些环境下使用dispose()或tidy()没有副作用。不过，主动调用通常会比垃圾回收的清理带来更好的性能。

### tfjs 安装

Web 端安装

	$ yarn add @tensorflow/tfjs
	
Node.js 使用 TensorFlow.js 原生 C++ 加速。**坑**：MAC OS上，安装时会对原生 C++ 依赖包进行下载编译，慢，执行 gyp 需要使用 python v2 环境。

	$ yarn add @tensorflow/tfjs-node
	
Node.js 使用 TensorFlow.js GPU 加速（ Linux Only）

	$ yarn add @tensorflow/tfjs-node-gpu 

### tfjs 加载

基于浏览器的版本，加载 @tensorflow/tfjs 使用 tensorflow.js。如果是在 Node.js 环境中使用，需要引入 `@tensorflow/tfjs-node` 或 `@tensorflow/tfjs-gpu`

	import * as tf from '@tensorflow/tfjs'

## 使用 Tensorflow.js 和 React 生成数据集

下面的代码引用自 `./src/components/curve/curve.tsx`

![curve-data](../images/dev-curve-1.png)

$$ y = a x^2 + b x + c $$

### 随机生成 a, b, c 三个参数

	const [sCurveParams, setCurveParams] = useState<number[] | string>(INIT_PARAMS)

	const genCurveParams = (): number[] => {
	    return tf.tidy(() => {
	        const params = tf.randomUniform([3], -10, 10).toInt()
	        return Array.from(params.dataSync())
	    })
	}
		
	const Curve = (): JSX.Element => {
		...
	    useEffect(() => {
	        const [a, b, c] = genCurveParams()
	        setCurveParams([a, b, c])
	    }, [])
    	...
    }
	    
* `genCurveParams` 是个单纯的本地功能函数，放在 Curve 之外；使用它的 `useEffect` 被放在 Curve 组件的前部位置，符合我们前面说的使用 Hooks 的规则。
* `genCurveParams` 通过 `tf.randomUniform` 生成了 -10 到 10 之间的三个随机数，取整之后，用作 a, b, c。tfjs 里面还有好几种用于生成随机数的方法，用起来非常容易，可以根据问题需要使用。
* 这段代码被封装在 tf.tidy 中，以及时回收不用的内存。
* useEffect 第二个参数设置为 `[]`，表示在组件加载时调用。

### 实现公式计算 & useCallback

    const calc = useCallback((x: tf.Tensor) => {
        return tf.tidy(() => {
            const [a, b, c] = sCurveParams
            // = a * x^2 + b * x + c
            return x.pow(2).mul(a).add(x.mul(b)).add(c)
        })
    }, [sCurveParams])

* tf.Tensor 提供了很多用于张量计算的函数，使用函数式编程的链式调用用起来也比较方便。需要注意的是，这种链式的调用仅仅与顺序有关，没有“先乘除，后加减”的计算符的优先级。
* 假如在此处使用普通的 JS 函数实现，每一次 Curve 组件渲染都会生成一个新的 calc 函数实例。
* `useCallback` 是我们所用到的第三类 React Hook。`useCallback` 会返回一个 memoized 的函数，用来对函数进行缓存，防止总是重复的生成新的函数。calc 函数被封装到了 `useCallback` 之后，只有当触发条件 [sCurveParams] 被修改时，才会触发回调函数 calc 发生改变，创建新实例。

### 训练集和测试集的生成

    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()
    ...
    
    useEffect(() => {
        logger('init data set ...')

        // train set
        const trainTensorX = tf.randomUniform([TOTAL_RECORD], -1, 1)
        const trainTensorY = calc(trainTensorX)
        setTrainSet({ xs: trainTensorX, ys: trainTensorY })

        // test set
        const testTensorX = tf.randomUniform([TEST_RECORD], -1, 1)
        const testTensorY = calc(testTensorX)
        setTestSet({ xs: testTensorX, ys: testTensorY })

        return () => {
            logger('Train Data Dispose')
            // Specify how to clean up after this effect:
            trainTensorX?.dispose()
            trainTensorY?.dispose()
            testTensorX?.dispose()
            testTensorY?.dispose()
        }
    }, [calc])
    
* 仅在当 calc 由于参数改变而发生改变时，才触发对于训练集和测试集的更新。
* 随机生成了 1000 个 (-1,1) 之间的浮点数，作为训练集 trainTensorX。随机生成了 200 个 (-1,1) 之间的浮点数，作为测试集 testTensorX。
* 初始化数据的 `useEffect` 函数和以前的用法相比，有了返回值。在 effect 中返回一个函数是 effect 的清除机制。每个 effect 都可以返回一个清除函数，它们都属于 effect 的一部分。对于 tfjs 应用来说，正好可以在这里清除不用的 tf.Tensor 对象，React Hooks 和 Tensorflow.js 真是相得益彰。React 会在执行当前 effect 之前对上一个 effect 进行清除。

		return () => {
            logger('Train Data Dispose')
            // Specify how to clean up after this effect:
            trainTensorX?.dispose()
            trainTensorY?.dispose()
            testTensorX?.dispose()
            testTensorY?.dispose()
        }

### 运用 AntD From 实现参数调整

AntD v4 的 Form 做了较大的修改，我们一起来看看。

	import React, { useCallback, useEffect, useRef, useState } from 'react'
	...
	import { Button, Card, Col, Form, Slider, Row, Select, Tabs } from 'antd'
	
	const Curve = (): JSX.Element => {
		...
		const [formData] = Form.useForm()
		...
		
		const handleResetCurveParams = (): void => {
	        const [a, b, c] = genCurveParams()
	        formData.setFieldsValue({ a, b, c })
	        setCurveParams([a, b, c])
	    }
	
	    const handleCurveParamsChange = (): void => {
	        const values = formData.getFieldsValue()
	        // logger('handleParamsFormChange', values)
	        const { a, b, c } = values
	        setCurveParams([a, b, c])
	    }
    
	    const curveParam = (): JSX.Element => {
	        return <Slider min={-10} max={10} marks={{ '-10': -10, 0: 0, 10: 10 }} />
	    }
	
	    const dataAdjustCard = (): JSX.Element => {
	        return (
	            <Card title='Adjust Data' style={{ margin: '8px' }} size='small'>
	                <Form {...layout} form={formData} onFieldsChange={handleCurveParamsChange}
	                    initialValues={{
	                        a: sCurveParams[0],
	                        b: sCurveParams[1],
	                        c: sCurveParams[2]
	                    }}>
	                    <Form.Item name='a' label='Curve param a'>
	                        {curveParam()}
	                    </Form.Item>
	                    ...
	                    <Form.Item {...tailLayout} >
	                        <Button onClick={handleResetCurveParams} style={{ width: '60%', margin: '0 20%' }}> Random a,b,c </Button>
	                        ...
	                    </Form.Item>
	                </Form>
	            </Card>
	        )
	    }
	    ...

* 在 Curve 组件的前部，使用 `const [formData] = Form.useForm()` 定义 Form 的数据域引用。`useForm` 只能用于函数组件。
* 在 Form 表单定义部分，使用 `form={formData}` 与数据域引用相关联。使用 `initialValues` 属性定义标点数据初始值。

		<Form {...layout} form={formData} onFieldsChange={handleCurveParamsChange}
	                    initialValues={{
	                        a: sCurveParams[0],
	                        b: sCurveParams[1],
	                        c: sCurveParams[2]
	                    }}>
	                    
* Form 内的各数据项使用 Form.Item 装饰。	其 `name`属性为 Form 内变量名称。
                 
		<Form.Item name='a' label='Curve param a'>
			{curveParam()}
		</Form.Item>
 
* 在界面上调整 Slider 组件时，会触发由 `onFieldsChange={handleCurveParamsChange}` 定义的回调函数。利用 `const values = formData.getFieldsValue()` 读取 Form 中的数据值。
* 点击 Button 时，`onClick={handleResetCurveParams}` 定义的回调函数会采用 `formData.setFieldsValue({ a, b, c })` 设置 From 中的数据值。

		<Button onClick={handleResetCurveParams} style={{ width: '60%', margin: '0 20%' }}> Random a,b,c </Button>
		
* 在 Form 中用 `onFinish` 函数设置 Form 的 Submit。

## 函数数据可视化

要对训练集和测试集数据进行直观的观察，我们使用了阿里巴巴的前端领域通用图表组件库 Bizchart。Bizchart 的功能相当强大，在这个项目中只使用了九牛一毛。[BizCharts参考链接](https://bizcharts.net/)

`/src/components/curve/CurveVis.tsx` 封装了函数曲线可视化的组件。

	<CurveVis xDataset={sTrainSet.xs as tf.Tensor} yDataset={sTrainSet.ys as tf.Tensor} 
		sampleCount={TOTAL_RECORD}/>

CurveVis 的实现要点如下：

	import React, { useEffect, useState } from 'react'
	...
	import { Axis, Chart, Geom, Legend, Tooltip } from 'bizcharts'
	
	import { arrayDispose, logger } from '../../utils'
	
	const MAX_SAMPLES_COUNT = 100
	...
	interface IChartData {
	    x: number
	    y: number
	    type: string
	}

	interface IProps {
	    xDataset: Tensor
	    yDataset: Tensor
	    pDataset?: Tensor
	    sampleCount?: number
	
	    debug?: boolean
	}
	
	const CurveVis = (props: IProps): JSX.Element => {

	    const [xData, setXData] = useState<number[]>([])
	    const [yData, setYData] = useState<number[]>([])
	    const [pData, setPData] = useState<number[]>([])
	    const [data, setData] = useState()
	    const [sampleCount] = useState(props.sampleCount)

	    ...
	    useEffect(() => {
	        logger('init sample data [p] ...')
	
	        const _data: IChartData[] = []
	        pData?.forEach((v: number, i: number) => {
	            _data.push({ x: xData[i], y: yData[i], type: 'y' })
	            _data.push({ x: xData[i], y: v, type: 'p' })
	        })
	        setData(_data)
	
	        return () => {
	            logger('Dispose sample data [p] ...')
	            arrayDispose(_data)
	        }
	    }, [pData])
	
	    return (
	        <Card>
	            <Chart height={400} data={data} padding='auto' forceFit>
	                <Axis name='X'/>
	                <Axis name='Y'/>
	                <Legend/>
	                <Tooltip/>
	                <Geom type='line' position='x*y' size={2} color={'type'} shape={'smooth'}/>
	            </Chart>
	            Sample count : {props.sampleCount}
	            {props.debug ? JSON.stringify(data) : ''}
	        </Card>
	    )
	}
	
	export default CurveVis
	
* 需要将从属性设置的 X、Y、P Tensor 转化成格式如 IChartData 的数组。使用 IChartData.type 区分不同的曲线。

		interface IChartData {
		    x: number
		    y: number
		    type: string
		} 
		
		...
	    const _data: IChartData[] = []
	    pData?.forEach((v: number, i: number) => {
	        _data.push({ x: xData[i], y: yData[i], type: 'y' })
	        _data.push({ x: xData[i], y: v, type: 'p' })
	    })
	    setData(_data)

* 使用如下方式绘制函数曲线。 

		return (
	        <Card>
	            <Chart height={400} data={data} padding='auto' forceFit>
	                <Axis name='X'/>
	                <Axis name='Y'/>
	                <Legend/>
	                <Tooltip/>
	                <Geom type='line' position='x*y' size={2} color={'type'} shape={'smooth'}/>
	            </Chart>
	            Sample count : {props.sampleCount}
	            {props.debug ? JSON.stringify(data) : ''}
	        </Card>
	    )
                   
## 使用 Tensorflow.js 创建人工神经网络

![curve-data](../images/dev-curve-2.png)

### 实现一个简单的多层人工神经网络

通过 formModel 调整，可以调整人工神经网络的层数 sLayerCount、每层的神经元数 sDenseUnits、以及激活函数 sActivation。当这几个值改变的时候，Curve.tsx 会相应调整人工神经网络模型。

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        // The linear regression model.
        const model = tf.sequential()
        model.add(tf.layers.dense({ inputShape: [1], units: sDenseUnits, activation: sActivation as any }))

        for (let i = sLayerCount - 2; i > 0; i--) {
            model.add(tf.layers.dense({ units: sDenseUnits, activation: sActivation as any }))
        }

        model.add(tf.layers.dense({ units: 1 }))
        setModel(model)

        return () => {
            logger('Model Dispose')
            model.dispose()
        }
    }, [sActivation, sLayerCount, sDenseUnits])
    
* 使用 tf.sequential 很容易构建出顺序多层神经网络。最简单的顺序全联接网络。tf.Sequential 是 LayerModel 的一个实例。

		const model = tf.sequential()
		
* 为网络增加输入层，因为 X 为一维向量，所以 `inputShape: [1]`。

		model.add(tf.layers.dense({ inputShape: [1], units: sDenseUnits, activation: sActivation as any }))

* 中间根据用户选择，增加多个隐藏层。

		for (let i = sLayerCount - 2; i > 0; i--) {
			model.add(tf.layers.dense({ units: sDenseUnits, activation: sActivation as any }))
		}
		
* 输出层，因为只输出一维的 Y 值，所以 `{ units: 1 }`。
	
		model.add(tf.layers.dense({ units: 1 }))

### 窥探一下 LayerModel 的内部

* 使用 `model.summary()`是最常用的观察模型的方法，不过只能够在浏览器的 Console 里显示结果。
* 实现一个简单的模型展示组件 `/src/components/common/tensor/ModelInfo.tsx`，看看模型的层次和权重相关的信息。

		import React from 'react'
		import * as tf from '@tensorflow/tfjs'
		
		interface IProps {
		    model: tf.LayersModel
		}
		
		const ModelInfo = (props: IProps): JSX.Element => {
		    const { model } = props
		    return (
		        <>
		            <div>
		                <h2>Layers</h2>
		                {model.layers.map((l, index) => <div key={index}>{l.name}</div>)}
		            </div>
		            <div>
		                <h2>Weights</h2>
		                {model.weights.map((w, index) => <div key={index}>{w.name}, [{w.shape.join(', ')}]</div>)}
		            </div>
		        </>
		    )
		}
		
		export default ModelInfo

## 模型训练 

![curve-data](../images/dev-curve-3.png)

在模型训练这个 Tab 中，我们可以对数据、模型、以及训练参数进行调整，以观察参数变化的影响。

### 调整 LearningRate 观察对训练的影响

    useEffect(() => {
        if (!sModel) {
            return
        }
        logger('init optimizer ...')

        const optimizer = tf.train.sgd(sLearningRate)
        sModel.compile({ loss: 'meanSquaredError', optimizer })

        return () => {
            logger('Optimizer Dispose')
            optimizer.dispose()
        }
    }, [sModel, sLearningRate])

* 调整 “随机梯度下降 SGD” 优化器的 sLearningRate，需要通过 `sModel.compile` 使之生效。

		const optimizer = tf.train.sgd(sLearningRate)
		sModel.compile({ loss: 'meanSquaredError', optimizer })

* 生成了新的优化器后，可以对老的优化器做清除。

		return () => {
			logger('Optimizer Dispose')
			optimizer.dispose()
		}

### 模型训练 model.fit

	const trainModel = (model: tf.LayersModel, trainSet: tf.TensorContainerObject, testSet: tf.TensorContainerObject): void => {
        if (!model || !trainSet || !testSet) {
            return
        }

        setStatus(STATUS.WAITING)
        stopRef.current = false

        model.fit(trainSet.xs as tf.Tensor, trainSet.ys as tf.Tensor, {
            epochs: NUM_EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: VALIDATE_SPLIT,
            callbacks: {
                onEpochEnd: async (epoch: number) => {
                    const trainStatus = `${(epoch + 1).toString()}/${NUM_EPOCHS.toString()} = ${((epoch + 1) / NUM_EPOCHS * 100).toFixed(0)} %`
                    setTrainStatusStr(trainStatus)

                    if (epoch % 10 === 0) {
                        evaluateModel(model, testSet)
                    }

                    if (stopRef.current) {
                        logger('Checked stop', stopRef.current)
                        setStatus(STATUS.STOPPED)
                        model.stopTraining = stopRef.current
                    }

                    await tf.nextFrame()
                }
            }
        }).then(
            () => {
                setStatus(STATUS.TRAINED)
            },
            loggerError
        )
    }

* `model.fit` 是模型训练的函数。训练时，还需要指定下面的参数。和 Python 不同，这些参数需要封装在 JS 对象中传递：
	* epochs 迭代次数
	* batchSize 因为计算环境资源有限，每次取用合适的数据量，以避免内存溢出等问题。
	* validationSplit 从训练集中挑选验证集数据的比率

	        model.fit(trainSet.xs as tf.Tensor, trainSet.ys as tf.Tensor, {
	            epochs: NUM_EPOCHS,
	            batchSize: BATCH_SIZE,
	            validationSplit: VALIDATE_SPLIT,
	            callbacks: {...}
	            })
            
* 一般来讲，训练一次会需要花费较长的时间。通过设置回调函数，我们能够及时了解训练过程的中间状态。`onEpochEnd` 函数在每个 Epoch 迭代结束时被调用，下面的代码展示的是，每 10 个 Epoch，使用当前模型里的 Weights 值，进行一次推理验证，并将结果推送出来。

				onEpochEnd: async (epoch: number) => {
                    const trainStatus = `${(epoch + 1).toString()}/${NUM_EPOCHS.toString()} = ${((epoch + 1) / NUM_EPOCHS * 100).toFixed(0)} %`
                    setTrainStatusStr(trainStatus)

                    if (epoch % 10 === 0) {
                        evaluateModel(model, testSet)
                    }
                    ...

                    await tf.nextFrame()
                }

### 及时停止模型训练 —— useRef Hook 登场

想中止训练，可以通过在 `model.stopTraining = true` 语句来完成。

我们在 `onEpochEnd` 增加了一段，试一下，看看这两段看起来功能相似的代码，执行结果有何不同？

	const stopRef = useRef(false)
	const [sStop, setStop] = useState<boolean>(false)
	...
	
				onEpochEnd: async (epoch: number) => {
                    ...
                    
						// Compare useRef with useState
                    if (sStop) {
                        logger('Checked stop by useState', sStop)
                        setStatus(STATUS.STOPPED)
                        model.stopTraining = sStop
                    }						
                    if (stopRef.current) {
                        logger('Checked stop by useRef', stopRef.current)
                        setStatus(STATUS.STOPPED)
                        model.stopTraining = stopRef.current
                    }

                    await tf.nextFrame()
                }

* 第一段实现使用了 `const [sStop, setStop] = useState<boolean>(false)`。为停止训练设置了 sStop，如果 sStop 为 true，则，停止训练。当用户点击相应按钮时，`setStop(true)`

	    const handleTrainStopState = (): void => {
	        logger('handleTrainStopState')
	        setStop(true)
	    }

* 第二段实现使用了 `const stopRef = useRef(false)`。当用户点击相应按钮时，`stopRef.current = true`

	    const handleTrainStop = (): void => {
	        logger('handleTrainStop')
	        stopRef.current = true
	    }

你实验出差别了吗？Why？

1. `setState` 用于状态的更新，state不能存储跨渲染周期的数据，因为state的保存会触发组件重渲染。`onEpochEnd` 函数在训练开始时被创建，它用到的是当时的 sStop 实例 sStop_1。而当 sStop 变化之后，触发了页面渲染，在新的渲染中，sStop 已经变成了一个新实例 sStop_2。此时，这就是为什么 setState 不起作用的原因。
2. 而 `useRef` 则返回一个可变的 ref 对象，返回的 ref 对象在组件的整个生命周期内保持不变。所以，Ref 可以用于在渲染周期之间共享数据的存储，对它修改也 **不会** 引起组件渲染。也就是说，stopRef.current 随时都指向的“那个”对象的当前值。

官方文档中关于 useRef 有个 setInterval 的例子，在一定程度上有利于理解这个问题。[Is there something like instance variables?](https://reactjs.org/docs/hooks-faq.html#is-there-something-like-instance-variables)

`useRef` 还被用于获取DOM元素的节点、获取子组件的实例、以及在渲染周期之间共享数据的存储等场景。
关于 useRef 的更多信息：[官方文档 useRef](https://reactjs.org/docs/hooks-reference.html#useref) 

## 模型推理

可以通过 model.predict 或 model.evaluate 来检验训练结果。

	const evaluateModel = (model: tf.LayersModel, testSet: tf.TensorContainerObject): void => {
        if (!model || !testSet) {
            return
        }

        const pred = model.predict(testSet.xs as tf.Tensor) as tf.Tensor
        setTestP(pred)
        const evaluate = model.evaluate(testSet.xs as tf.Tensor, testSet.ys as tf.Tensor) as tf.Scalar
        setTestV(evaluate)
    }
