import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs-core'
import { Button, Card, Col, Row } from 'antd'

import { ITrainInfo, logger, STATUS } from '../../utils'
import { MnistDatasetCore } from './dataCore'
import * as modelCore from './modelCore'

import SampleDataVis from '../common/tensor/SampleDataVis'
import TfvisHistoryWidget from '../common/tensor/TfvisHistoryWidget'
import TfvisDatasetInfoWidget from '../common/tensor/TfvisDatasetInfoWidget'

const MnistWeb = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)
    const [errors, setErrors] = useState()

    const [dataSet, setDataSet] = useState<MnistDatasetCore>()

    const [trainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [validSet, setValidSet] = useState<tf.TensorContainerObject>()

    const [logMsg, setLogMsg] = useState<ITrainInfo>()

    const [predictSet, setPredictSet] = useState<tf.TensorContainerObject>()
    const [predictResult, setPredictResult] = useState<tf.Tensor>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data set ...')
        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.LOADING)

        const mnistDataset = new MnistDatasetCore()
        mnistDataset.loadData().then(
            () => {
                setDataSet(mnistDataset)
                setTrainSet({ xs: mnistDataset.datasetImages, ys: mnistDataset.datasetLabels })
                setValidSet({ xs: mnistDataset.testImages, ys: mnistDataset.testLabels })

                setStatus(STATUS.LOADED)
            },
            () => {
                // ignore
            })
    }, [])

    useEffect(() => {
        logger('init predict data set ...')

        const testExamples = 50
        const batch = dataSet?.nextTestBatch(testExamples)

        const xTest = batch?.xs
        const yTest = batch?.labels as tf.Tensor

        const [ys] = tf.tidy(() => {
            const ys = yTest?.argMax(-1)
            return [ys]
        })
        setPredictSet({ xs: xTest, ys })
    }, [dataSet])

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

    const pushTrainingLog = (iteration: number, logs: any): void => {
        logs && addTrainInfo({ iteration: iteration, logs })
        predictModel(predictSet?.xs as tf.Tensor)
    }

    const trainModel = (_dataset: MnistDatasetCore): void => {
        if (!_dataset) {
            return
        }

        setStatus(STATUS.TRAINING)

        const beginMs = performance.now()
        modelCore.train(_dataset, pushTrainingLog).then(
            () => {
                setStatus(STATUS.TRAINED)

                const secSpend = (performance.now() - beginMs) / 1000
                logger(`Spend : ${secSpend.toString()}s`)
            },
            (error) => {
                setErrors(error)
            })
    }

    const predictModel = (_xs: tf.Tensor): void => {
        if (!_xs) {
            return
        }

        // const predictions = model.predict(batch.xs)
        // const labels = model.classesFromLabel(batch.labels)
        const preds = modelCore.predict(_xs)
        setPredictResult(preds)
    }

    const addTrainInfo = (info: ITrainInfo): void => {
        setLogMsg(info)
    }

    const handleTrain = (): void => {
        if (!dataSet) {
            return
        }
        // Train the model using the data.
        trainModel(dataSet)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>MNIST</h1>
            <Col span={12}>
                <Card title='Train' style={{ margin: '8px' }} size='small'>
                    <div style={{ color: 'red' }}>
                        !!! ATTENTION !!! Please go to ./public/data folder, run `download_data.sh`
                    </div>
                    <div>trainSet:
                        {dataSet && <TfvisDatasetInfoWidget value={{ xs: dataSet.datasetImages, ys: dataSet.datasetLabels }}/>}
                    </div>

                    <Button onClick={handleTrain} type='primary'> Train </Button>
                    <p>backend: {tfBackend}</p>
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
