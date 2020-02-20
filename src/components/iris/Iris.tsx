import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row } from 'antd'

import { arrayDispose, IDataSet, IModel, ITensor, ITrainInfo, IValidInfo, STATUS, logger } from '../../utils'

import * as data from './data'
import ModelInfo from '../common/tensor/ModelInfo'
import HistoryWidget from '../common/tensor/HistoryWidget'
import SampleDataVis from '../common/tensor/SampleDataVis'

const BATCH_SIZE = 32

const Iris = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)

    const [totalRecord] = useState<number>(10)
    const [testSplit] = useState<number>(0.15)
    const [trainSet, setTrainSet] = useState<IDataSet>()
    const [validSet, setValidSet] = useState<IDataSet>()

    const [totalEpochs] = useState<number>(40)
    const [learningRate] = useState<number>(0.01)
    const [model, setModel] = useState<IModel>()
    const [trainInfos, setTrainInfos] = useState<ITrainInfo[]>([])

    const [predictSet, setPredictSet] = useState<IValidInfo>()
    const [predictResult, setPredictResult] = useState<ITensor>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        const _model = tf.sequential()
        _model.add(tf.layers.dense({ units: 10, activation: 'sigmoid', inputShape: [data.IRIS_NUM_FEATURES] }))
        _model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))
        // _model.summary()

        const optimizer = tf.train.adam(learningRate)
        _model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

        setModel(_model)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [learningRate])

    useEffect(() => {
        logger('init data set ...')
        const [tSet, vSet] = data.getIrisData(testSplit)
        // Batch datasets.
        setTrainSet(tSet.batch(BATCH_SIZE))
        setValidSet(vSet.batch(BATCH_SIZE))

        return () => {
            logger('Train Data Dispose')
            // Specify how to clean up after this effect:
        }
    }, [testSplit, totalRecord])

    useEffect(() => {
        logger('init predict data set ...')
        validSet?.toArray().then(
            (result: any[]) => {
                const { xs: xTest, ys: yTest } = result[0]
                const [xs, ys] = tf.tidy(() => {
                    const xs = xTest
                    const ys = yTest.argMax(-1)
                    return [xs, ys]
                })
                setPredictSet({ xs, ys })
                // tf.dispose([xDataset, yDataset])
            },
            () => {
                // ignore
            })
    }, [validSet])

    /***********************
     * useEffects only for dispose
     ***********************/

    useEffect(() => {
        // Do Nothing

        return () => {
            logger('Predict Set Dispose')
            tf.dispose(predictSet?.xs)
            tf.dispose(predictSet?.ys)
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

    const trainModel = (_model: IModel, _trainDataset: IDataSet, _validDataset: IDataSet, options?: any): void => {
        if (!_model || !_trainDataset || !_validDataset) {
            return
        }

        const { epochs } = options

        setStatus(STATUS.TRAINING)
        resetTrainInfo()
        const beginMs = performance.now()
        // Call `model.fit` to train the model.
        _model.fitDataset(_trainDataset, {
            epochs: epochs || 40,
            validationData: _validDataset,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    logger('onEpochEnd', epoch)

                    // const secPerEpoch = (performance.now() - beginMs) / (1000 * (epoch + 1))
                    logs && addTrainInfo({ step: epoch, logs })
                    predictModel(_model, predictSet?.xs)
                }
            }
        }).then(
            () => {
                setStatus(STATUS.TRAINED)

                const secPerEpoch = (performance.now() - beginMs) / (1000 * epochs)
                logger(secPerEpoch)
            },
            () => {
                // ignore
            })
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
        // trainInfos.splice(0, trainInfos.length)
        arrayDispose(trainInfos)
        setTrainInfos([...trainInfos])
    }

    const handleTrain = (): void => {
        if (!model || !trainSet || !validSet) {
            return
        }
        // Train the model using the data.
        trainModel(model, trainSet, validSet, { epochs: totalEpochs })
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>鸢尾花分类 Iris</h1>
            <Col span={12}>
                <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                    <Button onClick={handleTrain} type='primary'> Train & Validate </Button>
                    <div>status: {status}</div>
                    <SampleDataVis xDataset={predictSet?.xs} yDataset={predictSet?.ys} pDataset={predictResult}
                        xFloatFixed={1} pageSize={18}/>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Model' style={{ margin: '8px' }} size='small'>
                    {model ? <ModelInfo model={model}/> : ''}
                    <div>backend: {tfBackend}</div>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Visualization' style={{ margin: '8px' }} size='small'>
                    <HistoryWidget infos={trainInfos} totalEpochs={totalEpochs} />
                </Card>
            </Col>
        </Row>
    )
}

export default Iris
