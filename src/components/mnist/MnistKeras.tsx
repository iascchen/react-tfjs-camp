import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Row, Select, Tabs } from 'antd'

import SampleDataVis from '../common/tensor/SampleDataVis'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import DrawPanelWidget from '../common/tensor/DrawPanelWidget'

import { layout, tailLayout } from '../../constant'
import { ILayerSelectOption, ITrainInfo, logger, STATUS } from '../../utils'
import { MnistGzDataset } from './dataGz'
import { MnistCoreDataset } from './dataCore'
import { addCovDropoutLayers, addDenseLayers, addCovPoolingLayers } from './model'
import TfvisHistoryWidget from '../common/tfvis/TfvisHistoryWidget'

const { Option } = Select
const { TabPane } = Tabs

// Data
const DATA_SOURCE = ['Web', 'Gz']
const BATCH_SIZES = [64, 128, 256, 512]
const SHOW_TEST_SAMPLE = 100

// Model
const MODELS = ['dense', 'cnn-pooling', 'cnn-dropout']

// Train
const EPOCHS = 5
const VALID_SPLIT = 0.15
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

const MnistKeras = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(5)

    // General
    const [sTfBackend, setTfBackend] = useState<string>()
    const statusRef = useRef<STATUS>(STATUS.INIT)
    const [sErrors, setErrors] = useState()

    // Data
    const [sDataSourceName, setDataSourceName] = useState()
    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()

    // const [sPredictSet, setPredictSet] = useState<tf.TensorContainerObject>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    // Model
    const [sModelName, setModelName] = useState('cnn-dropout')
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    // Train
    const [sLearningRate, setLearningRate] = useState<number>()
    const [sBatchSize, setBatchSize] = useState<number>()
    const stopRef = useRef(false)

    // Predict
    const [logMsg, setLogMsg] = useState<ITrainInfo>()
    const [sDrawPred, setDrawPred] = useState<tf.Tensor>()

    const [formTrain] = Form.useForm()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        // Create a sequential neural network model. tf.sequential provides an API
        // for creating "stacked" models where the output from one layer is used as
        // the input to the next layer.
        const model = tf.sequential()
        switch (sModelName) {
            case 'dense' :
                addDenseLayers(model)
                break
            case 'cnn-pooling' :
                addCovPoolingLayers(model)
                break
            case 'cnn-dropout' :
                addCovDropoutLayers(model)
                break
        }
        // Our last layer is a dense layer which has 10 output units, one for each
        // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
        // represent numbers, but it's the same idea if you had classes that
        // represented other entities like dogs and cats (two output classes: 0, 1).
        // We use the softmax function as the activation for the output layer as it
        // creates a probability distribution over our 10 classes so their output
        // values sum to 1.
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))
        setModel(model)

        const layerOptions: ILayerSelectOption[] = model?.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(layerOptions)

        return () => {
            logger('Model Dispose')
            model?.dispose()
        }
    }, [sModelName])

    useEffect(() => {
        if (!sModel) {
            return
        }
        logger('init model optimizer...', sLearningRate)

        const optimizer = tf.train.adam(sLearningRate)
        sModel.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
    }, [sModel, sLearningRate])

    useEffect(() => {
        logger('init data set ...')

        statusRef.current = STATUS.LOADING

        let mnistDataset: MnistGzDataset | MnistCoreDataset
        if (sDataSourceName === 'Gz') {
            mnistDataset = new MnistGzDataset()
        } else {
            mnistDataset = new MnistCoreDataset()
        }

        let tSet: tf.TensorContainerObject
        let vSet: tf.TensorContainerObject
        mnistDataset.loadData().then(
            () => {
                tSet = mnistDataset.getTrainData()
                vSet = mnistDataset.getTestData(SHOW_TEST_SAMPLE)

                setTrainSet(tSet)
                setTestSet(vSet)

                statusRef.current = STATUS.LOADED
            },
            (error) => {
                logger(error)
            }
        )

        return () => {
            logger('Data Set Dispose')
            tf.dispose([tSet.xs, tSet.ys])
            tf.dispose([vSet.xs, vSet.ys])
        }
    }, [sDataSourceName])

    /***********************
     * useEffects only for dispose
     ***********************/

    useEffect(() => {
        // Do Nothing
        return () => {
            logger('Predict Result Dispose')
            sPredictResult?.dispose()
        }
    }, [sPredictResult])

    /***********************
     * Functions
     ***********************/

    const predictModel = (model: tf.LayersModel, xs: tf.Tensor): void => {
        if (!model || !xs) {
            return
        }
        const preds = tf.tidy(() => {
            return model.predict(xs) as tf.Tensor
        })
        setPredictResult(preds)
    }

    const trainModel = (model: tf.LayersModel, trainDataset: tf.TensorContainerObject, validDataset: tf.TensorContainerObject): void => {
        if (!model || !trainDataset || !validDataset) {
            return
        }

        statusRef.current = STATUS.TRAINING
        stopRef.current = false

        const beginMs = performance.now()

        let trainBatchCount = 0
        let iteration = 0
        model.fit(trainDataset.xs as tf.Tensor, trainDataset.ys as tf.Tensor, {
            epochs: EPOCHS,
            batchSize: sBatchSize,
            validationSplit: VALID_SPLIT,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    logger('onEpochEnd', epoch)

                    logs && addTrainInfo({ iteration: iteration++, logs })
                    predictModel(model, validDataset.xs as tf.Tensor)

                    await tf.nextFrame()
                },
                onBatchEnd: async (batch, logs) => {
                    trainBatchCount++
                    logs && addTrainInfo({ iteration: iteration++, logs })
                    if (batch % 50 === 0) {
                        logger(`onBatchEnd: ${batch.toString()} / ${trainBatchCount.toString()}`)
                        predictModel(model, validDataset.xs as tf.Tensor)
                    }
                    await tf.nextFrame()

                    if (stopRef.current) {
                        logger('Checked stop', stopRef.current)
                        statusRef.current = STATUS.STOPPED
                        model.stopTraining = stopRef.current
                    }
                }
            }
        }).then(
            () => {
                statusRef.current = STATUS.TRAINED

                const secSpend = (performance.now() - beginMs) / 1000
                logger(`Spend : ${secSpend.toString()}s`)
            },
            (error) => {
                statusRef.current = STATUS.STOPPED
                setErrors(error)
            }
        )
    }

    const evaluateModel = (_model: tf.LayersModel, _validDataset: tf.TensorContainerObject): void => {
        if (!_model || !_validDataset) {
            return
        }
        const evalOutput = _model.evaluate(_validDataset.xs as tf.Tensor, _validDataset.ys as tf.Tensor) as tf.Tensor[]
        logger(`Final evaluate Loss: ${evalOutput[0].dataSync()[0].toFixed(3)} %`)
        logger(`Final evaluate Accuracy: ${evalOutput[1].dataSync()[0].toFixed(3)} %`)
    }

    const addTrainInfo = (info: ITrainInfo): void => {
        setLogMsg(info)
    }

    const handleEvaluate = (): void => {
        if (!sModel || !sTestSet) {
            return
        }
        // Evaluate the model using the data.
        evaluateModel(sModel, sTestSet)
    }

    const handleDataSourceChange = (value: string): void => {
        setDataSourceName(value)
    }

    const handleModelChange = (value: string): void => {
        setModelName(value)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleDrawSubmit = (data: tf.Tensor): void => {
        if (!sModel) {
            return
        }
        // logger('handleDrawSubmit', data.shape)
        const pred = tf.tidy(() => sModel.predict(data)) as tf.Tensor
        // logger('handleDrawSubmit', pred.dataSync())
        setDrawPred(pred)
    }

    const handleBatchSizeChange = (value: number): void => {
        setBatchSize(value)
    }

    const handleLearningRateChange = (value: number): void => {
        // logger('handleLearningRateChange', value)
        setLearningRate(value)
    }

    const handleTrain = (): void => {
        if (!sModel || !sTrainSet || !sTestSet) {
            return
        }
        // Train the model using the data.
        trainModel(sModel, sTrainSet, sTestSet)
    }

    const handleTrainStop = (): void => {
        logger('handleTrainStop')
        stopRef.current = true
    }

    const handleWeightSave = (): void => {
        logger('handleWeightSave')
        // stopRef.current = true
    }

    const handleWeightLoad = (): void => {
        logger('handleWeightLoad')
        // stopRef.current = true
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    const dataAdjustCard = (): JSX.Element => {
        return (
            <Card title='Data Source' style={{ margin: '8px' }} size='small'>
                <Form {...layout} initialValues={{
                    dataSource: 'Web'
                }}>
                    <Form.Item name='dataSource' label='Select Data Source'>
                        <Select onChange={handleDataSourceChange}>
                            {DATA_SOURCE.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <div>由于数据量较大，多次加载会影响程序运行效率。</div>
                        <div style={{ color: 'red' }}>!!! 请注意 !!! 如果您是从 Github 上克隆项目，在运行之前，
                            请先前往目录 ./public/data , 运行 download_mnist.sh 脚本，下载所需的数据。</div>
                        <div>如果您是在 Docker 中运行，数据已经预先放在相应的目录下。</div>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Model' style={{ margin: '8px' }} size='small'>
                <Form {...layout} initialValues={{
                    modelName: 'cnn-dropout'
                }}>
                    <Form.Item name='modelName' label='Select Model'>
                        <Select onChange={handleModelChange}>
                            {MODELS.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    const trainAdjustCard = (): JSX.Element => {
        return (
            <Card title='Train' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formTrain} onFinish={handleTrain} initialValues={{
                    learningRate: 0.001,
                    batchSize: 256
                }}>
                    <Form.Item name='learningRate' label='Learning Rate'>
                        <Select onChange={handleLearningRateChange}>
                            {LEARNING_RATES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='batchSize' label='Batch Size'>
                        <Select onChange={handleBatchSizeChange} >
                            {BATCH_SIZES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button type='primary' htmlType={'submit'} style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                        <Button onClick={handleTrainStop} style={{ width: '30%', margin: '0 10%' }}> Stop </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <div>backend: {sTfBackend}</div>
                        <div>Status: {statusRef.current}</div>
                        <div>Errors: {sErrors}</div>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button onClick={handleWeightSave} style={{ width: '30%', margin: '0 10%' }}> Save </Button>
                        <Button onClick={handleWeightLoad} style={{ width: '30%', margin: '0 10%' }}> Load </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button onClick={handleEvaluate} style={{ width: '30%', margin: '0 10%' }}> Evaluate </Button>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    return (
        <AIProcessTabs title={'MNIST LayerModel'} current={sTabCurrent} onChange={handleTabChange} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/mnist.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={8}>
                        {dataAdjustCard()}
                    </Col>
                    <Col span={16}>
                        <Card title='Train Data Set' style={{ margin: '8px' }} size='small'>
                            <div>{sTrainSet && <TfvisDatasetInfoWidget value={sTrainSet}/>}</div>
                            {/* <SampleDataVis xDataset={sTrainSet?.xs as tf.Tensor} yDataset={sTrainSet?.ys as tf.Tensor} */}
                            {/*    xIsImage pageSize={10}/> */}
                        </Card>
                        <Card title={`Validate Data Set (Only show ${SHOW_TEST_SAMPLE} samples)`} style={{ margin: '8px' }} size='small'>
                            <div>{sTestSet && <TfvisDatasetInfoWidget value={sTestSet}/>}</div>
                            <SampleDataVis xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}
                                xIsImage pageSize={10}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={8}>
                        {modelAdjustCard()}
                        <Card title='Show Layer' style={{ margin: '8px' }} size='small'>
                            <Form {...layout} initialValues={{
                                layer: 0
                            }}>
                                <Form.Item name='layer' label='Show Layer'>
                                    <Select onChange={handleLayerChange} >
                                        {sLayersOption?.map((v) => {
                                            return <Option key={v.index} value={v.index}>{v.name}</Option>
                                        })}
                                    </Select>
                                </Form.Item>
                            </Form>
                        </Card>
                    </Col>
                    <Col span={16}>
                        <Card title='Model Info' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                        </Card>
                        <Card title='Layer Info' style={{ margin: '8px' }} size='small'>
                            <TfvisLayerWidget layer={sCurLayer}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={6}>
                        {trainAdjustCard()}
                        {modelAdjustCard()}
                    </Col>
                    <Col span={10}>
                        <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                            <SampleDataVis xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}
                                pDataset={sPredictResult} xIsImage pageSize={10}/>
                        </Card>
                    </Col>
                    <Col span={8}>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <TfvisHistoryWidget logMsg={logMsg} debug />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Col span={8}>
                    <DrawPanelWidget onSubmit={handleDrawSubmit} prediction={sDrawPred} />
                </Col>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MnistKeras
