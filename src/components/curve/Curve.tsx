import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Slider, Row, Select, Tabs } from 'antd'
import MathJax from '@matejmazur/react-mathjax'

import { layout, tailLayout } from '../../constant'
import { logger, loggerError, STATUS } from '../../utils'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import ModelInfo from '../common/tensor/ModelInfo'

import CurveVis from './CurveVis'

const { Option } = Select
const { TabPane } = Tabs

// data
const TOTAL_RECORD = 1000
const TEST_RECORD = TOTAL_RECORD * 0.2
const INIT_PARAMS = [0, 1, 0]

// model
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
const ACTIVATIONS = ['sigmoid', 'relu', 'tanh']

// train
const NUM_EPOCHS = 200
const BATCH_SIZE = 100
const VALIDATE_SPLIT = 0.2

const genCurveParams = (): number[] => {
    const params = tf.randomUniform([3], -10, 10).toInt()
    return Array.from(params.dataSync())
}

const numberToStringWithSign = (x: number): string => {
    const signStr = Math.sign(x) >= 0 ? '+' : ''
    return `${signStr}${x}`
}

const Curve = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    // General
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>()

    // Data
    const [sCurveParams, setCurveParams] = useState<number[] | string>(INIT_PARAMS)

    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()

    const [sTestP, setTestP] = useState<tf.Tensor>()
    const [sTestV, setTestV] = useState<tf.Scalar>()

    // Model
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sActivation, setActivation] = useState<string>('sigmoid')
    const [sLayerCount, setLayerCount] = useState<number>(2)
    const [sDenseUnits, setDenseUnits] = useState<number>(4)

    // Train
    const [sLearningRate, setLearningRate] = useState<number>(0.03)
    const [sTrainStatusStr, setTrainStatusStr] = useState<string>('0%')
    const stopRef = useRef(false)

    const [formData] = Form.useForm()
    const [formModel] = Form.useForm()

    /***********************
     * useCallback
     ***********************/

    const calc = useCallback((x: tf.Tensor) => {
        const [a, b, c] = sCurveParams
        // = a * x^2 + b * x + c
        return x.pow(2).mul(a).add(x.mul(b)).add(c)
    }, [sCurveParams])

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        const [a, b, c] = genCurveParams()
        setCurveParams([a, b, c])
    }, [])

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

    /***********************
     * Functions
     ***********************/

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

    const evaluateModel = (model: tf.LayersModel, testSet: tf.TensorContainerObject): void => {
        if (!model || !testSet) {
            return
        }

        const pred = model.predict(testSet.xs as tf.Tensor) as tf.Tensor
        setTestP(pred)
        const evaluate = model.evaluate(testSet.xs as tf.Tensor, testSet.ys as tf.Tensor) as tf.Scalar
        setTestV(evaluate)
    }

    const handlePredict = (): void => {
        if (!sModel || !sTestSet) {
            return
        }
        evaluateModel(sModel, sTestSet)
    }

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

    const handleModelParamsChange = (): void => {
        const values = formModel.getFieldsValue()
        // logger('handleSuperParamsChange', values)
        const { layers, activation, units } = values
        setLayerCount(layers)
        setActivation(activation)
        setDenseUnits(units)
    }

    const handleLearningRateChange = (value: string): void => {
        // logger('handleLearningRateChange', value)
        setLearningRate(+value)
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

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

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
                    <Form.Item name='b' label='b'>
                        {curveParam()}
                    </Form.Item>
                    <Form.Item name='c' label='c'>
                        {curveParam()}
                    </Form.Item>
                    <Form.Item {...tailLayout} >
                        <Button onClick={handleResetCurveParams} style={{ width: '60%', margin: '0 20%' }}> Random a,b,c </Button>
                        <div className='centerContainer' style={{ margin: '16px' }}>
                            <MathJax.Context>
                                <MathJax.Node>{`y = ${sCurveParams[0]} x^2 ${numberToStringWithSign(sCurveParams[1] as number)} x ${numberToStringWithSign(sCurveParams[2] as number)}`}</MathJax.Node>
                            </MathJax.Context>
                        </div>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    const trainDataCard = (): JSX.Element => {
        if (!sTrainSet) {
            return <></>
        }
        return (
            <Card title='Train Data Set' style={{ margin: '8px' }} size='small'>
                <CurveVis xDataset={sTrainSet.xs as tf.Tensor} yDataset={sTrainSet.ys as tf.Tensor}
                    sampleCount={TOTAL_RECORD}/>
            </Card>
        )
    }

    const testDataCard = (): JSX.Element => {
        if (!sTestSet) {
            return <></>
        }
        return (
            <Card title='Test Data Set' style={{ margin: '8px' }} size='small'>
                <CurveVis xDataset={sTestSet.xs as tf.Tensor} yDataset={sTestSet.ys as tf.Tensor} pDataset={sTestP}
                    sampleCount={TEST_RECORD}/>
                <div className='centerContainer' style={{ margin: '16px' }}>
                    <Button type='primary' onClick={handlePredict}> Validate </Button>
                </div>
                <div>Trained epoches: {sTrainStatusStr} </div>
                <div>Evaluate loss: {sTestV?.dataSync().join(' , ')}</div>
            </Card>
        )
    }

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Model' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formModel} onFieldsChange={handleModelParamsChange} initialValues={{
                    layers: 2,
                    units: 4,
                    activation: 'sigmoid'
                }}>
                    <Form.Item name='layers' label='Layer Count'>
                        <Slider min={2} max={5} marks={{ 2: 2, 5: 5 }}/>
                    </Form.Item>
                    <Form.Item name='units' label='Unit'>
                        <Slider min={4} max={8} step={2} marks={{ 4: 4, 6: 6, 8: 8 }}/>
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
                <Form {...layout} onFinish={handleTrain} initialValues={{
                    learningRate: 0.03
                }}>
                    <Form.Item name='learningRate' label='Learning Rate'>
                        <Select onChange={handleLearningRateChange}>
                            {LEARNING_RATES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
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
        <AIProcessTabs title={'曲线拟合 Curve'} current={sTabCurrent} onChange={handleTabChange}
            invisiblePanes={[AIProcessTabPanes.PREDICT]}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/curve.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        {dataAdjustCard()}
                    </Col>
                    <Col span={12}>
                        {trainDataCard()}
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
                        {trainDataCard()}
                    </Col>
                    <Col span={9}>
                        {testDataCard()}
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default Curve
