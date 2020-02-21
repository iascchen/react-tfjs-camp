import React, { useCallback, useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row, Select } from 'antd'

import { STATUS, logger, range } from '../../utils'
import ModelInfo from '../common/tensor/ModelInfo'
import CurveVis from './CurveVis'

const { Option } = Select

// model
const LayersCount = ['1', '2', '3']
const Activations = ['sigmoid', 'relu']

// data
const TOTAL_RECORD = 1000
const TEST_RECORD = TOTAL_RECORD * 0.2
const VALIDATE_SPLIT = 0.2
const INIT_PARAMS = [0, 1, 0]

// train
const NUM_EPOCHS = 200
const BATCH_SIZE = 50
const LEARNING_RATE = 0.05

const Curve = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    // General
    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)

    // Model
    const [model, setModel] = useState<tf.LayersModel>()
    const [activation, setActivation] = useState('sigmoid')
    const [layerCount, setLayerCount] = useState('2')

    // Data
    const [curveParams, setCurveParams] = useState<number[]>(INIT_PARAMS)
    const [totalRecord] = useState<number>(TOTAL_RECORD)
    const [testRecord] = useState<number>(TEST_RECORD)

    const [trainX, setTrainX] = useState<tf.Tensor>()
    const [trainY, setTrainY] = useState<tf.Tensor>()
    const [testX, setTestX] = useState<tf.Tensor>()
    const [testY, setTestY] = useState<tf.Tensor>()
    const [testP, setTestP] = useState<tf.Tensor>()
    const [testV, setTestV] = useState<tf.Scalar>()

    // Train
    const [trainStatusStr, setTrainStatusStr] = useState<string>('0%')

    /***********************
     * useCallback
     ***********************/

    const calc = useCallback((x: tf.Tensor) => {
        const [a, b, c] = curveParams

        // = a * x^2 + b * x + c
        return (x.mul(a).mul(x)).add(x.mul(b)).add(c)

        // // = a * x^2 + b * x + c
        // return x.mul(b).add(c)
    }, [curveParams])

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        const params = range(0, 3).map(() => Math.round(Math.random() * 10))
        setCurveParams(params)
    }, [])

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        // The linear regression model.
        const _model = tf.sequential()

        switch (layerCount) {
            case '1' :
                _model.add(tf.layers.dense({ inputShape: [1], units: 1 }))
                break
            case '2' :
                _model.add(tf.layers.dense({ inputShape: [1], units: 4, activation: activation as any }))
                _model.add(tf.layers.dense({ units: 1 }))
                break
            case '3' :
                _model.add(tf.layers.dense({ inputShape: [1], units: 4, activation: activation as any }))
                _model.add(tf.layers.dense({ units: 4, activation: activation as any }))
                _model.add(tf.layers.dense({ units: 1 }))
                break
        }

        const optimizer = tf.train.sgd(LEARNING_RATE)
        _model.compile({ loss: 'meanSquaredError', optimizer })
        // _model.summary()

        setModel(_model)

        return () => {
            logger('Model Dispose')
            _model.dispose()
        }
    }, [activation, layerCount])

    useEffect(() => {
        logger('init data set ...')

        // train set
        const _trainTensorX = tf.randomUniform([totalRecord], -1, 1)
        const _trainTensorY = calc(_trainTensorX)
        setTrainX(_trainTensorX)
        setTrainY(_trainTensorY)

        // test set
        const _testTensorX = tf.randomUniform([testRecord], -1, 1)
        const _testTensorY = calc(_testTensorX)
        setTestX(_testTensorX)
        setTestY(_testTensorY)

        return () => {
            logger('Train Data Dispose')
            // Specify how to clean up after this effect:
            _trainTensorX.dispose()
            _trainTensorY.dispose()
            _testTensorX.dispose()
            _testTensorY.dispose()
        }
    }, [totalRecord, testRecord, calc])

    /***********************
     * Functions
     ***********************/

    const trainModel = (_model: tf.LayersModel, _trainX: tf.Tensor, _trainY: tf.Tensor): void => {
        if (!_model || !_trainX || !_trainY) {
            return
        }

        setStatus(STATUS.TRAINING)
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        _model.fit(_trainX, _trainY, {
            epochs: NUM_EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: VALIDATE_SPLIT,
            callbacks: {
                onEpochEnd: (epoch: number) => {
                    const trainStatus = `${(epoch + 1).toString()}/${NUM_EPOCHS.toString()} = ${((epoch + 1) / NUM_EPOCHS * 100).toFixed(0)} %`
                    setTrainStatusStr(trainStatus)

                    if (epoch % 10 === 0) {
                        evaluateModel(_model, testX, testY)
                    }
                }
            }
        }).then(
            () => {
                // Use the model to do inference on a data point the model hasn't seen before:
                setStatus(STATUS.TRAINED)
            },
            () => {
                // ignore
            })
    }

    const evaluateModel = (_model: tf.LayersModel, _testX: tf.TensorContainer, _testY: tf.TensorContainer): void => {
        if (!_model || !_testX || !_testY) {
            return
        }

        const pred = _model.predict(_testX as tf.Tensor) as tf.Tensor
        setTestP(pred)
        const evaluate = _model.evaluate(_testX as tf.Tensor, _testY as tf.Tensor) as tf.Scalar
        setTestV(evaluate)
    }

    const handleTrain = (): void => {
        if (!model || !trainX || !trainY) {
            return
        }
        // Train the model using the data.
        trainModel(model, trainX, trainY)
    }

    const handlePredict = (): void => {
        if (!model || !testX || !testY) {
            return
        }
        evaluateModel(model, testX, testY)
    }

    const handleResetParams = (): void => {
        const params = range(0, 3).map(() => Math.floor(Math.random() * 10))
        // logger('a,b,c', params.join(', '))
        setCurveParams(params)
    }

    const handleActivateChange = (value: string): void => {
        setActivation(value)
    }

    const handleLayerCountChange = (value: string): void => {
        setLayerCount(value)
    }

    return (
        <Row gutter={16}>
            <h1>曲线拟合 Curve</h1>
            <Col span={24}>
                <Card title='Model' style={{ margin: '8px' }} size='small'>
                    {model ? <ModelInfo model={model}/> : ''}
                    <div>Backend: {tfBackend}</div>
                    <div>
                        Layer Count :
                        <Select onChange={handleLayerCountChange} defaultValue={'2'}>
                            {LayersCount.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </div>
                    <div>
                        Activation :
                        <Select onChange={handleActivateChange} defaultValue={'sigmoid'}>
                            {Activations.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </div>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Train' style={{ margin: '8px' }} size='small'>
                    {/* <p>trainX: {trainX?.dataSync().join(' , ')}</p> */}
                    {/* <p>trainY: {trainY?.dataSync().join(' , ')}</p> */}
                    <CurveVis xDataset={trainX} yDataset={trainY} sampleCount={totalRecord} />

                    <div>
                        Curve Params: {`${curveParams[0].toString()}*x^2 + ${curveParams[1].toString()}*x + ${curveParams[2].toString()}`}
                        <Button onClick={handleResetParams}> Reset </Button>
                    </div>
                    <div>
                        Status: {status}
                        <Button type='primary' onClick={handleTrain}> Train </Button>
                    </div>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Predict' style={{ margin: '8px' }} size='small'>
                    <CurveVis xDataset={testX} yDataset={testY} pDataset={testP} sampleCount={testRecord} />
                    <Button type='primary' onClick={handlePredict}> Validate </Button>
                    <div>trained epoches: {trainStatusStr} </div>
                    <div>evaluate loss: {testV?.dataSync().join(' , ')}</div>
                </Card>
            </Col>
        </Row>
    )
}

export default Curve
