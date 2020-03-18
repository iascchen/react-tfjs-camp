import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Row, Select, Slider, Tabs } from 'antd'

import { arrayDispose, IDataSet, ITrainInfo, logger, loggerError, STATUS } from '../../utils'

import { layout, tailLayout } from '../../constant'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import ModelInfo from '../common/tensor/ModelInfo'
import HistoryWidget from '../common/tensor/HistoryWidget'
import SampleDataVis from '../common/tensor/SampleDataVis'
import MarkdownWidget from '../common/MarkdownWidget'

import * as data from './data'

const { Option } = Select
const { TabPane } = Tabs

const LABEL = 'Label'
const ONE_HOT = 'One-Hot'
const TARGET_ENCODE = [LABEL, ONE_HOT]

const EPOCHS = 40
const BATCH_SIZE = 32
const VALIDATE_SPLIT = 0.15

const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
const ACTIVATIONS = ['sigmoid', 'relu', 'tanh']
const OPTIMIZERS = ['SGD', 'Adam', 'RMSProp']

const Iris = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>()

    // Data
    const [sTargetEncode, setTargetEncode] = useState(ONE_HOT)

    const [sTrainSet, setTrainSet] = useState<IDataSet>()
    const [sValidSet, setValidSet] = useState<IDataSet>()

    const [sPredictSet, setPredictSet] = useState<tf.TensorContainerObject>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    // model
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sActivation, setActivation] = useState<string>('sigmoid')
    const [sDenseUnits, setDenseUnits] = useState<number>(10)

    // Train
    const [sLearningRate, setLearningRate] = useState<number>(0.01)
    const [sOptimizer, setOptimizer] = useState<string>('Adam')
    const [sLoss, setLoss] = useState<string>('categoricalCrossentropy')
    const stopRef = useRef(false)

    const [sTrainInfos, setTrainInfos] = useState<ITrainInfo[]>([])

    const [formModel] = Form.useForm()
    const [formTrain] = Form.useForm()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        if (!sTargetEncode) {
            return
        }
        logger('encode dataset ...')

        const [tSet, vSet] = data.getIrisData(VALIDATE_SPLIT, sTargetEncode === ONE_HOT)

        // Batch datasets.
        setTrainSet(tSet.batch(BATCH_SIZE))
        setValidSet(vSet.batch(BATCH_SIZE))

        // one-hot or int-label data, use different loss
        const loss = sTargetEncode === ONE_HOT ? 'categoricalCrossentropy' : 'sparseCategoricalCrossentropy'
        setLoss(loss)

        return () => {
            logger('Encode Data Dispose')
        }
    }, [sTargetEncode])

    useEffect(() => {
        if (!sLearningRate || !sLoss || !sDenseUnits) {
            return
        }
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        const model = tf.sequential()
        model.add(tf.layers.dense({
            units: sDenseUnits,
            activation: sActivation as any,
            inputShape: [data.IRIS_NUM_FEATURES]
        }))
        model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))
        setModel(model)

        return () => {
            logger('Model Dispose')
            model?.dispose()
        }
    }, [sActivation, sDenseUnits])

    useEffect(() => {
        if (!sModel) {
            return
        }
        logger('init optimizer ...')

        let optimizer: tf.Optimizer
        switch (sOptimizer) {
            case 'SGD' :
                optimizer = tf.train.sgd(sLearningRate)
                break
            case 'RMSProp' :
                optimizer = tf.train.rmsprop(sLearningRate)
                break
            case 'Adam' :
            default:
                optimizer = tf.train.adam(sLearningRate)
                break
        }

        sModel.compile({ optimizer: optimizer, loss: sLoss, metrics: ['accuracy'] })
        // setModel(model)

        return () => {
            logger('Optimizer Dispose')
            optimizer?.dispose()
        }
    }, [sModel, sLearningRate, sOptimizer, sLoss])

    useEffect(() => {
        logger('init predict data set ...')
        sValidSet?.toArray().then(
            (result: any[]) => {
                // const { xs: xTest, ys: yTest } = result[0]
                setPredictSet(result[0])
            },
            loggerError
        )
    }, [sValidSet])

    /***********************
     * useEffects only for dispose
     ***********************/

    useEffect(() => {
        // Do Nothing

        return () => {
            logger('Predict Set Dispose')
            tf.dispose(sPredictSet?.xs)
            tf.dispose(sPredictSet?.ys)
        }
    }, [sPredictSet])

    useEffect(() => {
        return () => {
            logger('Predict Result Dispose')
            sPredictResult?.dispose()
        }
    }, [sPredictResult])

    /***********************
     * Functions
     ***********************/

    const trainModel = (model: tf.LayersModel, trainDataset: IDataSet, validDataset: IDataSet): void => {
        if (!model || !trainDataset || !validDataset) {
            return
        }

        setStatus(STATUS.TRAINING)
        stopRef.current = false
        resetTrainInfo()

        const beginMs = performance.now()
        model.fitDataset(trainDataset, {
            epochs: EPOCHS,
            validationData: validDataset,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    logger('onEpochEnd', epoch)

                    logs && addTrainInfo({ iteration: epoch, logs })
                    predictModel(model, sPredictSet?.xs)
                },
                onBatchEnd: () => {
                    if (stopRef.current) {
                        logger('onBatchEnd Checked stop', stopRef.current)
                        setStatus(STATUS.STOPPED)
                        model.stopTraining = stopRef.current
                    }
                }
            }
        }).then(
            () => {
                setStatus(STATUS.TRAINED)

                const secPerEpoch = (performance.now() - beginMs) / (1000 * EPOCHS)
                logger(secPerEpoch)
            },
            loggerError
        )
    }

    const predictModel = (_model: tf.LayersModel, _xs: tf.TensorContainer): void => {
        if (!_model || !_xs) {
            return
        }
        const [preds] = tf.tidy(() => {
            const preds = _model.predict(_xs as tf.Tensor) as tf.Tensor
            return [preds]
        })
        setPredictResult(preds)
    }

    const addTrainInfo = (info: ITrainInfo): void => {
        sTrainInfos.push(info)
        setTrainInfos([...sTrainInfos])
    }

    const resetTrainInfo = (): void => {
        logger('resetTrainInfo')
        arrayDispose(sTrainInfos)
        setTrainInfos([...sTrainInfos])
    }

    const handleTargetEncodeChange = (value: string): void => {
        // logger('handleTargetEncodeChange', value)
        setTargetEncode(value)
    }

    const handleModelParamsChange = (): void => {
        const values = formModel.getFieldsValue()
        // logger('handleSuperParamsChange', values)
        const { activation, units } = values
        setActivation(activation)
        setDenseUnits(units)
    }

    const handleTrainParamsChange = (): void => {
        const values = formTrain.getFieldsValue()
        // logger('handleTrainParamsChange', value)
        const { learningRate, optimizer } = values
        setLearningRate(learningRate)
        setOptimizer(optimizer)
    }

    const handleTrain = (): void => {
        if (!sModel || !sTrainSet || !sValidSet) {
            return
        }
        trainModel(sModel, sTrainSet, sValidSet)
    }

    const handleTrainStop = (): void => {
        // logger('handleTrainStop')
        stopRef.current = true
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    const dataAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Data' style={{ margin: '8px' }} size='small'>
                <Form {...layout} initialValues={{ encode: ONE_HOT }}>
                    <Form.Item name='encode' label='Target Encode'>
                        <Select onChange={handleTargetEncodeChange}>
                            {TARGET_ENCODE.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item label='Loss should be'>
                        <span>{sLoss}</span>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Model' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formModel} onFieldsChange={handleModelParamsChange} initialValues={{
                    units: 10,
                    activation: 'sigmoid'
                }}>
                    <Form.Item name='units' label='Units'>
                        <Slider min={4} max={12} step={2} marks={{ 4: 4, 8: 8, 12: 12 }}/>
                    </Form.Item>
                    <Form.Item name='activation' label='Activation'>
                        <Select>
                            {ACTIVATIONS.map((v) => {
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
                <Form {...layout} form={formTrain} onFinish={handleTrain} onFieldsChange={handleTrainParamsChange}
                    initialValues={{
                        learningRate: 0.01,
                        optimizer: 'Adam'
                    }}>
                    <Form.Item name='learningRate' label='Learning Rate'>
                        <Select>
                            {LEARNING_RATES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='optimizer' label='Optimizer'>
                        <Select>
                            {OPTIMIZERS.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item label='Loss'>
                        <span>{sLoss}</span>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button type='primary' htmlType={'submit'} style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                        <Button onClick={handleTrainStop} style={{ width: '30%', margin: '0 10%' }}> Stop </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <div>Status: {sStatus}</div>
                        <div>Backend: {sTfBackend}</div>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    return (
        <AIProcessTabs title={'鸢尾花分类 Iris'} current={sTabCurrent} onChange={handleTabChange}
            invisiblePanes={[AIProcessTabPanes.PREDICT]}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/iris.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        {dataAdjustCard()}
                    </Col>
                    <Col span={12}>
                        <Card title='Sample Data' style={{ margin: '8px' }} size='small'>
                            <SampleDataVis xDataset={sPredictSet?.xs as tf.Tensor}
                                yDataset={sPredictSet?.ys as tf.Tensor} xFloatFixed={1} pageSize={10}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        {modelAdjustCard()}
                    </Col>
                    <Col span={12}>
                        <Card title='Model' style={{ margin: '8px' }} size='small'>
                            {sModel ? <ModelInfo model={sModel}/> : ''}
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={6}>
                        {trainAdjustCard()}
                        {dataAdjustCard()}
                        {modelAdjustCard()}
                    </Col>
                    <Col span={9}>
                        <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                            <SampleDataVis xDataset={sPredictSet?.xs as tf.Tensor}
                                yDataset={sPredictSet?.ys as tf.Tensor} pDataset={sPredictResult}
                                xFloatFixed={1} pageSize={10}/>
                        </Card>
                    </Col>
                    <Col span={9}>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <HistoryWidget infos={sTrainInfos} totalIterations={EPOCHS}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default Iris
