import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row, Select } from 'antd'

import { ITrainInfo, logger, STATUS } from '../../utils'
import { addCnnLayers, addDenseLayers } from './model'
import { MnistWebDataset } from './data'

import SampleDataVis from '../common/tensor/SampleDataVis'
import TfvisHistoryWidget from '../common/tensor/TfvisHistoryWidget'
import TfvisModelWidget from '../common/tensor/TfvisModelWidget'
import TfvisLayerWidget from '../common/tensor/TfvisLayerWidget'
import TfvisDatasetInfoWidget from '../common/tensor/TfvisDatasetInfoWidget'

const { Option } = Select

const EPOCHS = 10
const BATCH_SIZE = 320
const VALID_SPLIT = 0.15
const LEARNING_RATE = 0.1

const Models = ['dense', 'cnn']

interface ILayerSelectOption {
    name: string
    index: number
}

const MnistWeb = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)
    const [errors, setErrors] = useState()

    const [trainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [validSet, setValidSet] = useState<tf.TensorContainerObject>()

    const [modelName, setModelName] = useState('dense')
    const [model, setModel] = useState<tf.LayersModel>()
    const [layersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()
    const [logMsg, setLogMsg] = useState<ITrainInfo>()

    const [predictSet, setPredictSet] = useState<tf.TensorContainerObject>()
    const [predictResult, setPredictResult] = useState<tf.Tensor>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        const _model = tf.sequential()
        switch (modelName) {
            case 'dense' :
                addDenseLayers(_model)
                break
            case 'cnn' :
                addCnnLayers(_model)
                break
        }

        const optimizer = tf.train.sgd(LEARNING_RATE)
        _model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
        setModel(_model)

        const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(_layerOptions)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [modelName])

    useEffect(() => {
        logger('init data set ...')

        setStatus(STATUS.LOADING)
        const mnistDataset = new MnistWebDataset()

        let tSet: tf.TensorContainerObject
        let vSet: tf.TensorContainerObject
        mnistDataset.loadData().then(
            () => {
                tSet = mnistDataset.getTrainData()
                vSet = mnistDataset.getTestData()

                setTrainSet(tSet)
                setValidSet(vSet)

                setStatus(STATUS.LOADED)
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
    }, [])

    useEffect(() => {
        logger('init predict data set ...')

        const xTest = validSet?.xs as tf.Tensor
        const yTest = validSet?.ys as tf.Tensor

        const [ys] = tf.tidy(() => {
            const ys = yTest?.argMax(-1)
            return [ys]
        })
        setPredictSet({ xs: xTest, ys })
    }, [validSet])

    /***********************
     * useEffects only for dispose
     ***********************/

    // useEffect(() => {
    //     // Do Nothing
    //
    //     return () => {
    //         logger('Train Set Dispose')
    //         tf.dispose([trainSet?.xs, trainSet?.ys])
    //     }
    // }, [trainSet])
    //
    // useEffect(() => {
    //     // Do Nothing
    //
    //     return () => {
    //         logger('Valid Set Dispose')
    //         tf.dispose([validSet?.xs, validSet?.ys])
    //     }
    // }, [validSet])

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

    const trainModel = (_model: tf.LayersModel, _trainDataset: tf.TensorContainerObject): void => {
        if (!_model || !_trainDataset) {
            return
        }

        setStatus(STATUS.TRAINING)

        // We'll keep a buffer of loss and accuracy values over time.
        let trainBatchCount = 0
        const beginMs = performance.now()

        // Call `model.fit` to train the model.
        let iteration = 0
        _model.fit(_trainDataset.xs as tf.Tensor, _trainDataset.ys as tf.Tensor, {
            epochs: EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: VALID_SPLIT,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    logger('onEpochEnd', epoch)

                    // const secPerEpoch = (performance.now() - beginMs) / (1000 * (epoch + 1))
                    logs && addTrainInfo({ iteration: iteration++, logs })
                    predictModel(_model, predictSet?.xs as tf.Tensor)

                    await tf.nextFrame()
                },
                onBatchEnd: (batch, logs) => {
                    trainBatchCount++
                    logs && addTrainInfo({ iteration: iteration++, logs })
                    if (batch % 50 === 0) {
                        logger(`onBatchEnd: ${batch.toString()} / ${trainBatchCount.toString()}`)
                        predictModel(_model, predictSet?.xs as tf.Tensor)
                    }
                    // await tf.nextFrame()
                }
            }
        }).then(
            () => {
                setStatus(STATUS.TRAINED)

                const secSpend = (performance.now() - beginMs) / 1000
                logger(`Spend : ${secSpend.toString()}s`)
            },
            (error) => {
                setErrors(error)
            })
    }

    const evaluateModel = (_model: tf.LayersModel, _validDataset: tf.TensorContainerObject): void => {
        if (!_model || !_validDataset) {
            return
        }
        const evalOutput = _model.evaluate(_validDataset.xs as tf.Tensor, _validDataset.ys as tf.Tensor) as tf.Tensor[]
        logger(`Final evaluate Loss: ${evalOutput[0].dataSync()[0].toFixed(3)} %`)
        logger(`Final evaluate Accuracy: ${evalOutput[1].dataSync()[0].toFixed(3)} %`)
    }

    const predictModel = (_model: tf.LayersModel, _xs: tf.Tensor): void => {
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
        setLogMsg(info)
    }

    const handleTrain = (): void => {
        if (!model || !trainSet) {
            return
        }
        // Train the model using the data.
        trainModel(model, trainSet)
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

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = model?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>MNIST</h1>
            <Col span={12}>
                <Card title='Model' style={{ margin: '8px' }} size='small'>
                    <div>
                        Select Model : <Select onChange={handleModelChange} defaultValue={'dense'}>
                            {Models.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                        <TfvisModelWidget model={model}/>
                    </div>
                    <div>
                        Select Layer : <Select onChange={handleLayerChange} defaultValue={0}>
                            {layersOption?.map((v) => {
                                return <Option key={v.index} value={v.index}>{v.name}</Option>
                            })}
                        </Select>
                        <TfvisLayerWidget layer={curLayer}/>
                    </div>

                    <p>backend: {tfBackend}</p>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Train' style={{ margin: '8px' }} size='small'>
                    <div style={{ color: 'red' }}>!!! ATTENTION !!! Please go to ./public/data folder, run `download_data.sh`</div>
                    <div>trainSet: {trainSet && <TfvisDatasetInfoWidget value={trainSet}/>}</div>
                    <div>validSet: {validSet && <TfvisDatasetInfoWidget value={validSet}/>}</div>

                    <Button onClick={handleTrain} type='primary'> Train </Button>
                    <Button onClick={handleEvaluate}> Evaluate </Button>
                    <p>status: {status}</p>
                    <p>errors: {errors}</p>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Visualization' style={{ margin: '8px' }} size='small'>
                    <TfvisHistoryWidget logMsg={logMsg} debug />
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                    <SampleDataVis xDataset={predictSet?.xs as tf.Tensor} yDataset={predictSet?.ys as tf.Tensor}
                        pDataset={predictResult} xIsImage />
                </Card>
            </Col>
        </Row>
    )
}

export default MnistWeb
