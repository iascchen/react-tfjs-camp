import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, InputNumber, Row, Select, Tabs } from 'antd'
import MathJax from '@matejmazur/react-mathjax'

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
const VALIDATE_SPLIT = 0.2
const INIT_PARAMS = [0, 1, 0]

// model
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
const ACTIVATIONS = ['sigmoid', 'relu']

// train
const NUM_EPOCHS = 200
const BATCH_SIZE = 50

const layout = {
    labelCol: { span: 8 },
    wrapperCol: { span: 16 }
}

const genCurveParams = (): number[] => {
    const params = tf.randomUniform([3], 0, 10).toInt()
    return Array.from(params.dataSync())
}

const Curve = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    // General
    const [sTfBackend, setTfBackend] = useState<string>()
    const statusRef = useRef<STATUS>(STATUS.INIT)

    // Model
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sActivation, setActivation] = useState<string>('sigmoid')
    const [sLayerCount, setLayerCount] = useState<number>(2)
    const [sDenseUnits, setDenseUnits] = useState<number>(4)

    // Data
    const [sCurveParams, setCurveParams] = useState<number[] | string>(INIT_PARAMS)

    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()

    const [sTestP, setTestP] = useState<tf.Tensor>()
    const [sTestV, setTestV] = useState<tf.Scalar>()

    // Train
    const [sLearningRate, setLearningRate] = useState<number>(0.03)
    const [sTrainStatusStr, setTrainStatusStr] = useState<string>('0%')
    const stopRef = useRef(false)

    const [formData] = Form.useForm()
    const [formModel] = Form.useForm()
    const [formTrain] = Form.useForm()

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
        const _model = tf.sequential()

        switch (sLayerCount) {
            case 1 :
                _model.add(tf.layers.dense({ inputShape: [1], units: 1 }))
                break
            case 2 :
                _model.add(tf.layers.dense({ inputShape: [1], units: sDenseUnits, activation: sActivation as any }))
                _model.add(tf.layers.dense({ units: 1 }))
                break
            case 3 :
            case 4 :
            case 5 :
                _model.add(tf.layers.dense({ inputShape: [1], units: sDenseUnits, activation: sActivation as any }))
                for (let i = sLayerCount - 2; i > 0; i--) {
                    _model.add(tf.layers.dense({ units: sDenseUnits, activation: sActivation as any }))
                }
                _model.add(tf.layers.dense({ units: 1 }))
                break
        }
        setModel(_model)

        return () => {
            logger('Model Dispose')
            _model.dispose()
        }
    }, [sActivation, sLayerCount, sDenseUnits])

    useEffect(() => {
        if (!sModel) {
            return
        }
        logger('init model compile ...')

        const optimizer = tf.train.sgd(sLearningRate)
        sModel.compile({ loss: 'meanSquaredError', optimizer })
    }, [sModel, sLearningRate])

    useEffect(() => {
        logger('init data set ...')

        // train set
        const _trainTensorX = tf.randomUniform([TOTAL_RECORD], -1, 1)
        const _trainTensorY = calc(_trainTensorX)
        setTrainSet({ xs: _trainTensorX, ys: _trainTensorY })

        // test set
        const _testTensorX = tf.randomUniform([TEST_RECORD], -1, 1)
        const _testTensorY = calc(_testTensorX)
        setTestSet({ xs: _testTensorX, ys: _testTensorY })

        return () => {
            logger('Train Data Dispose')
            // Specify how to clean up after this effect:
            _trainTensorX?.dispose()
            _trainTensorY?.dispose()
            _testTensorX?.dispose()
            _testTensorY?.dispose()
        }
    }, [calc])

    /***********************
     * Functions
     ***********************/

    const trainModel = (model: tf.LayersModel, trainSet: tf.TensorContainerObject, testSet: tf.TensorContainerObject): void => {
        if (!model || !trainSet || !testSet) {
            return
        }

        statusRef.current = STATUS.TRAINING
        stopRef.current = false

        model.fit(trainSet.xs as tf.Tensor, trainSet.ys as tf.Tensor, {
            epochs: NUM_EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: VALIDATE_SPLIT,
            callbacks: {
                onEpochEnd: (epoch: number) => {
                    const trainStatus = `${(epoch + 1).toString()}/${NUM_EPOCHS.toString()} = ${((epoch + 1) / NUM_EPOCHS * 100).toFixed(0)} %`
                    setTrainStatusStr(trainStatus)

                    if (epoch % 10 === 0) {
                        evaluateModel(model, testSet)
                    }

                    if (stopRef.current) {
                        logger('Checked stop', stopRef.current)
                        statusRef.current = STATUS.STOPPED
                        model.stopTraining = stopRef.current
                    }
                }
            }
        }).then(
            () => {
                // Use the model to do inference on a data point the model hasn't seen before:
                statusRef.current = STATUS.TRAINED
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

    const handleModelChange = (): void => {
        const values = formModel.getFieldsValue()
        // logger('handleSuperParamsChange', values)
        const { layers, activation, units } = values
        setLayerCount(layers)
        setActivation(activation)
        setDenseUnits(units)
    }

    const handleTrain = (values: any): void => {
        const { learningRate } = values
        setLearningRate(learningRate)

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

    const dataAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Data' style={{ margin: '8px' }} size='small'>
                <Row className='centerContainer' style={{ margin: '8px' }}>
                    <Form {...layout} layout='inline' form={formData} onFieldsChange={handleCurveParamsChange}
                        initialValues={{
                            a: sCurveParams[0],
                            b: sCurveParams[1],
                            c: sCurveParams[2]
                        }}>
                        <Form.Item name='a' label='a'><InputNumber/></Form.Item>
                        <Form.Item name='b' label='b'><InputNumber/></Form.Item>
                        <Form.Item name='c' label='c'><InputNumber/></Form.Item>
                    </Form>
                </Row>
                <Row className='centerContainer' style={{ margin: '24px' }}>
                    <Button onClick={handleResetCurveParams}> Random a,b,c </Button>
                </Row>
                <Row className='centerContainer' style={{ margin: '24px' }}>
                    <MathJax.Context>
                        <MathJax.Node>{`y = ${sCurveParams[0]} x^2 + ${sCurveParams[1]} x + ${sCurveParams[2]}`}</MathJax.Node>
                    </MathJax.Context>
                </Row>
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
                <div className='centerContainer' style={{ margin: '16px'}}><Button type='primary' onClick={handlePredict}> Validate </Button></div>
                <div>trained epoches: {sTrainStatusStr} </div>
                <div>evaluate loss: {sTestV?.dataSync().join(' , ')}</div>
            </Card>
        )
    }

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Model' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formModel} onFieldsChange={handleModelChange} initialValues={{
                    layers: 2,
                    units: 4,
                    activation: 'sigmoid'
                }}>
                    <Form.Item name='layers' label='Layer Count'>
                        <InputNumber min={1} max={5}/>
                    </Form.Item>
                    <Form.Item name='units' label='Unit'>
                        <InputNumber min={4}/>
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
                <Row>
                    <Col span={12}>
                        <Form {...layout} layout='inline' form={formTrain} onFinish={handleTrain} initialValues={{
                            learningRate: 0.03
                        }}>
                            <Form.Item name='learningRate' label='Learning Rate'>
                                <Select>
                                    {LEARNING_RATES.map((v) => {
                                        return <Option key={v} value={v}>{v}</Option>
                                    })}
                                </Select>
                            </Form.Item>
                            <Form.Item>
                                <Button type='primary' htmlType={'submit'}> Train </Button>
                            </Form.Item>
                            <Form.Item>
                                <Button onClick={handleTrainStop}> Stop </Button>
                            </Form.Item>
                        </Form>
                    </Col>
                    <Col span={12}>
                        Status: {statusRef.current}
                    </Col>
                </Row>
            </Card>
        )
    }

    return (
        <AIProcessTabs title={'曲线拟合 Curve'} current={sTabCurrent} onChange={handleTabChange}>
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
                            <div>Backend: {sTfBackend}</div>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={12}>
                        {dataAdjustCard()}
                    </Col>
                    <Col span={12}>
                        {modelAdjustCard()}
                    </Col>
                    <Col span={24}>
                        {trainAdjustCard()}
                    </Col>
                    <Col span={12}>
                        {trainDataCard()}
                    </Col>
                    <Col span={12}>
                        {testDataCard()}
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Col span={12}>
                    {testDataCard()}
                </Col>
            </TabPane>
        </AIProcessTabs>
    )
}

export default Curve
