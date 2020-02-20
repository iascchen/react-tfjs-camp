import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row, Select } from 'antd'

import { ITrainDataSet, IModel, ITensor, ITrainInfo, STATUS, logger } from '../../utils'
import { MnistDataset } from './data'
import { addDenseLayers, addCnnLayers } from './model'
import ModelInfo from '../common/tensor/ModelInfo'
import HistoryWidget from '../common/tensor/HistoryWidget'
import SampleDataVis from '../common/tensor/SampleDataVis'

const { Option } = Select

const EPOCHS = 10
const BATCH_SIZE = 500
const VALID_SPLIT = 0.15

const Models = ['dense', 'cnn']

const Mnist = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)
    const [errors, setErrors] = useState()

    const [trainSet, setTrainSet] = useState<ITrainDataSet>()
    const [validSet, setValidSet] = useState<ITrainDataSet>()

    const [totalEpochs] = useState<number>(EPOCHS)

    const [modelName, setModelName] = useState('dense')
    const [model, setModel] = useState<IModel>()
    const [trainInfos, setTrainInfos] = useState<ITrainInfo[]>([])

    const [predictSet, setPredictSet] = useState<ITrainDataSet>()
    const [predictResult, setPredictResult] = useState<ITensor>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        // const _model = createDenseModel()
        const _model = tf.sequential()

        switch (modelName) {
            case 'dense' :
                addDenseLayers(_model)
                break
            case 'cnn' :
                addCnnLayers(_model)
                break
        }

        const optimizer = tf.train.adam(0.5)
        _model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

        setModel(_model)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [modelName])

    useEffect(() => {
        logger('init data set ...')

        setStatus(STATUS.LOADING)
        const mnistDataset = new MnistDataset()
        mnistDataset.loadData().then(
            () => {
                const tSet = mnistDataset.getTrainData()
                const eSet = mnistDataset.getTestData()

                setTrainSet(tSet)
                setValidSet(eSet)

                setStatus(STATUS.LOADED)
            },
            (e) => {
                setErrors(e)
            }
        )
    }, [])

    useEffect(() => {
        logger('init predict data set ...')

        const xTest = validSet?.xs as tf.Tensor
        const yTest = validSet?.ys as tf.Tensor

        const [xs, ys] = tf.tidy(() => {
            const xs = xTest
            const ys = yTest?.argMax(-1)
            return [xs, ys]
        })
        setPredictSet({ xs, ys })
    }, [validSet])

    /***********************
     * useEffects only for dispose
     ***********************/

    useEffect(() => {
        // Do Nothing

        return () => {
            logger('Train Set Dispose')
            tf.dispose([trainSet?.xs, trainSet?.ys])
        }
    }, [trainSet])

    useEffect(() => {
        // Do Nothing

        return () => {
            logger('Valid Set Dispose')
            tf.dispose([validSet?.xs, validSet?.ys])
        }
    }, [validSet])

    useEffect(() => {
        // Do Nothing

        return () => {
            logger('Predict Set Dispose')
            tf.dispose([predictSet?.xs, predictSet?.ys])
        }
    }, [predictSet])

    useEffect(() => {
        // Do Nothing
        return () => {
            logger('Predict Result Dispose')
            predictResult?.dispose()
        }
    }, [predictResult])

    /***********************
     * Functions
     ***********************/

    const trainModel = (_model: IModel, _trainDataset: ITrainDataSet, options?: any): void => {
        if (!_model || !_trainDataset) {
            return
        }

        const { epochs, batchSize, validationSplit } = options

        setStatus(STATUS.TRAINING)
        resetTrainInfo()

        // We'll keep a buffer of loss and accuracy values over time.
        let trainBatchCount = 0
        const beginMs = performance.now()

        // logger('trainDataset', _trainDataset.xDataset.dataSync(), _trainDataset.yDataset.dataSync())

        // Call `model.fit` to train the model.
        _model.fit(_trainDataset.xs, _trainDataset.ys, {
            epochs: epochs || EPOCHS,
            batchSize: batchSize || BATCH_SIZE,
            validationSplit: validationSplit || VALID_SPLIT,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    logger('onEpochEnd', epoch)

                    // const secPerEpoch = (performance.now() - beginMs) / (1000 * (epoch + 1))
                    logs && addTrainInfo({ step: epoch, logs })
                    predictModel(_model, predictSet?.xs)
                    // await tf.nextFrame()
                },
                onBatchEnd: (batch) => {
                    trainBatchCount++
                    if (batch % 50 === 0) {
                        logger(`onBatchEnd: ${batch.toString()} / ${trainBatchCount.toString()}`)
                        predictModel(_model, predictSet?.xs)
                    }
                }
            }
        }).then(
            () => {
                setStatus(STATUS.TRAINED)

                const secSpend = (performance.now() - beginMs) / 1000
                logger(`Spend : ${secSpend.toString()}`)
            },
            (error) => {
                setErrors(error)
            })
    }

    const evaluateModel = (_model: IModel, _validDataset: ITrainDataSet): void => {
        if (!_model || !_validDataset) {
            return
        }
        const evalOutput = _model.evaluate(_validDataset.xs, _validDataset.ys) as tf.Tensor[]
        logger(`Final evaluate Loss: ${evalOutput[0].dataSync()[0].toFixed(3)} %`)
        logger(`Final evaluate Accuracy: ${evalOutput[1].dataSync()[0].toFixed(3)} %`)
    }

    const predictModel = (_model: IModel, _xs: ITensor): void => {
        if (!_model || !_xs) {
            return
        }

        const [preds] = tf.tidy(() => {
            const preds = (_model.predict(_xs) as tf.Tensor).argMax(-1)
            return [preds]
        })
        setPredictResult(preds)
    }

    const addTrainInfo = (info: ITrainInfo): void => {
        // logger('addTrainInfo', trainInfos, info)
        trainInfos.push(info)
        setTrainInfos([...trainInfos])
    }

    const resetTrainInfo = (): void => {
        logger('resetTrainInfo')
        trainInfos.splice(0, trainInfos.length)
        setTrainInfos([...trainInfos])
    }

    const handleTrain = (): void => {
        if (!model || !trainSet) {
            return
        }
        // Train the model using the data.
        trainModel(model, trainSet, { epochs: totalEpochs })
    }

    const handleEvaluate = (): void => {
        if (!model || !validSet) {
            return
        }
        // Evaluate the model using the data.
        evaluateModel(model, validSet)
    }

    const handleModelChange = (value: string): void => {
        setModelName(value)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>MNIST</h1>
            <Col span={12}>
                <Card title='Model' style={{ margin: '8px' }} size='small'>
                    Select Model : <Select onChange={handleModelChange} defaultValue={'dense'}>
                        {Models.map((v) => {
                            return <Option key={v} value={v}>{v}</Option>
                        })}
                    </Select>

                    {model ? <ModelInfo model={model}/> : ''}
                    <p>backend: {tfBackend}</p>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Train' style={{ margin: '8px' }} size='small'>
                    <p>!!! ATTENTION !!! Please go to ./public/data folder, run `download_data.sh`</p>
                    <p>trainSet: {JSON.stringify(trainSet)}</p>
                    <p>validSet: {JSON.stringify(validSet)}</p>

                    <Button onClick={handleTrain}> Train </Button>
                    <Button onClick={handleEvaluate}> Evaluate </Button>
                    <p>status: {status}</p>
                    <p>errors: {errors}</p>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Visualization' style={{ margin: '8px' }} size='small'>
                    <HistoryWidget infos={trainInfos} totalEpochs={totalEpochs} />
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                    <SampleDataVis xDataset={predictSet?.xs} yDataset={predictSet?.ys} pDataset={predictResult} xIsImage />
                </Card>
            </Col>
        </Row>
    )
}

export default Mnist
