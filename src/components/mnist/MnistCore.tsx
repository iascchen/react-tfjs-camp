import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs-core'
import { Button, Card, Col, Row } from 'antd'

import { ITrainInfo, logger, STATUS } from '../../utils'
import { MnistCoreDataset } from './dataCore'
import * as modelCore from './modelCore'

import SampleDataVis from '../common/tensor/SampleDataVis'
import TfvisHistoryWidget from '../common/tensor/TfvisHistoryWidget'
import TfvisDatasetInfoWidget from '../common/tensor/TfvisDatasetInfoWidget'
import DrawPanelWidget from './DrawPanelWidget'

const MnistWeb = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)
    const [errors, setErrors] = useState()

    const [dataSet, setDataSet] = useState<MnistCoreDataset>()

    const [trainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [validSet, setValidSet] = useState<tf.TensorContainerObject>()

    const [logMsg, setLogMsg] = useState<ITrainInfo>()

    const [predictSet, setPredictSet] = useState<tf.TensorContainerObject>()
    const [predictResult, setPredictResult] = useState<tf.Tensor>()

    const [drawPred, setDrawPred] = useState<tf.Tensor>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data set ...')
        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.LOADING)

        const mnistDataset = new MnistCoreDataset()

        let tSet: tf.TensorContainerObject
        let vSet: tf.TensorContainerObject
        mnistDataset.loadData().then(
            () => {
                setDataSet(mnistDataset)

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

    const pushTrainingLog = (iteration: number, loss: number): void => {
        addTrainInfo({ iteration: iteration, logs: { loss } })
        predictModel(predictSet?.xs as tf.Tensor)
    }

    const trainModel = (_dataset: MnistCoreDataset): void => {
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

    const handleDrawSubmit = (data: tf.Tensor): void => {
        // logger('handleDrawSubmit', data.shape)
        const pred = modelCore.predict(data)
        logger('handleDrawSubmit', pred.dataSync())
        setDrawPred(pred)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>MNIST</h1>
            <Col span={12}>
                <Card title='Train' style={{ margin: '8px' }} size='small'>
                    <div style={{ color: 'red' }}>!!! ATTENTION !!! Please go to ./public/data folder, run `download_data.sh`</div>
                    <div>trainSet: {trainSet && <TfvisDatasetInfoWidget value={trainSet}/>}</div>
                    <div>validSet: {validSet && <TfvisDatasetInfoWidget value={validSet}/>}</div>

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
            <Col span={12}>
                <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                    <DrawPanelWidget onChange={handleDrawSubmit} prediction={drawPred} />
                </Card>
            </Col>
        </Row>
    )
}

export default MnistWeb
