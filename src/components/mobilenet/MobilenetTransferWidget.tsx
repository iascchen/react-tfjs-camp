import React, { useEffect, useReducer, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, message, Row, Select } from 'antd'

import {
    arrayDispose,
    ILabeledImage,
    ILabeledImageFileJson,
    ILabeledImageSet, ILabelMap,
    ILayerSelectOption,
    logger,
    STATUS
} from '../../utils'
import { MOBILENET_IMAGE_SIZE } from '../../constant'
import WebCamera, { IWebCameraHandler } from '../common/tensor/WebCamera'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import { createModel, createTruncatedMobileNet } from './modelTransfer'
import { TransferDataset } from './dataTransfer'
import LabeledCaptureInputSet from '../common/tensor/LabeledCaptureInputSet'
import LabeledCaptureSetWidget from '../common/tensor/LabeledCaptureSetWidget'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select

const MobilenetTransferWidget = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sOutputClasses, setOutputClasses] = useState<number>(4)
    const [sLearningRate, setLearningRate] = useState<number>(0.0001)
    const [sDenseUnits, setDenseUnits] = useState<number>(100)

    const [sTruncatedModel, setTruncatedModel] = useState<tf.LayersModel>()
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sBatchSize, setBatchSize] = useState<number>(0.4)
    const [sEpochs, setEpochs] = useState<number>(10)

    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()

    const [sImgUid, genImgUid] = useReducer((x: number) => x + 1, 0)

    const [sLabelsMap, setLabelsMap] = useState<ILabelMap>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    const webcamRef = useRef<IWebCameraHandler>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init truncatedModel model ...')

        setStatus(STATUS.LOADING)

        tf.backend()
        setTfBackend(tf.getBackend())

        let _truncatedModel: tf.LayersModel
        createTruncatedMobileNet().then(
            (result) => {
                _truncatedModel = result
                setTruncatedModel(result)
                setStatus(STATUS.LOADED)
            },
            (e) => {
                logger(e)
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                message.error(e.message)
            }
        )

        return () => {
            logger('TruncatedModel Dispose')
            _truncatedModel?.dispose()
        }
    }, [])

    useEffect(() => {
        logger('init model ...')
        if (!sTruncatedModel) {
            return
        }

        setStatus(STATUS.LOADING)

        const _model = createModel(sTruncatedModel, sOutputClasses, sLearningRate, sDenseUnits)
        setModel(_model)

        const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(_layerOptions)

        setStatus(STATUS.LOADED)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [sTruncatedModel, sOutputClasses, sLearningRate, sDenseUnits])

    useEffect(() => {
        if (!sLabeledImgs || !sTruncatedModel) {
            return
        }
        logger('init data set ...')

        const outputClasses = sLabeledImgs.length
        setOutputClasses(outputClasses)

        const labelsArray = sLabeledImgs.map((labeled) => labeled.label)
        const labelsMap: ILabelMap = {}
        labelsArray.forEach((item, index) => {
            labelsMap[index] = item
        })
        setLabelsMap(labelsMap)

        const dataHandler = new TransferDataset(outputClasses)
        dataHandler.addExamples(sTruncatedModel, sLabeledImgs)
        setTrainSet(() => dataHandler.getData()) // when use sTrainSet, will get last records

        return () => {
            logger('Data Dispose')
            dataHandler.dispose()
        }
    }, [sLabeledImgs])

    /***********************
     * Functions
     ***********************/

    const train = (_trainSet: tf.TensorContainerObject): void => {
        logger('train', _trainSet)
        if (!sModel) {
            return
        }

        setStatus(STATUS.TRAINING)

        // We parameterize batch size as a fraction of the entire dataset because the
        // number of examples that are collected depends on how many examples the user
        // collects. This allows us to have a flexible batch size.
        const _tensorX = _trainSet.xs as tf.Tensor
        const _tensorY = _trainSet.ys as tf.Tensor
        const batchSize = Math.floor(_tensorX.shape[0] * sBatchSize)
        if (!(batchSize > 0)) {
            throw new Error('Batch size is 0 or NaN. Please choose a non-zero fraction.')
        }

        const surface = { name: 'Logs', tab: 'Train Logs' }
        // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
        sModel.fit(_tensorX, _tensorY, {
            batchSize,
            epochs: sEpochs,
            callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc', 'val_loss', 'val_acc'])
        }).then(
            () => {
                setStatus(STATUS.TRAINED)
            },
            () => {
                // ignore
            })
    }

    const handleTrain = (): void => {
        sTrainSet && train(sTrainSet)
    }

    const handleLoadModel = (): void => {
        // TODO : Load saved model
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        message.info('TODO: Not Implemented')
    }

    const handlePredict = (imgTensor: tf.Tensor): void => {
        if (!imgTensor) {
            return
        }
        setStatus(STATUS.PREDICTING)
        // console.log('handlePredict', imgTensor)
        const [imgFeature] = tf.tidy(() => {
            const batched = imgTensor.reshape([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])
            const embeddings = sTruncatedModel?.predict(batched)
            const result = sModel?.predict(embeddings as tf.Tensor) as tf.Tensor
            const imgFeature = result.argMax(-1)
            return [imgFeature]
        })
        logger('Predict', imgFeature)
        setStatus(STATUS.PREDICTED)
        setPredictResult(imgFeature)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleLabeledImagesSubmit = (value: ILabeledImageFileJson): void => {
        logger('handleLabeledImagesSubmit', value)

        const labeledImageSetList = value.labeledImageSetList
        setLabeledImgs(labeledImageSetList)
    }

    const handleLoadJson = (values: ILabeledImageSet[]): void => {
        sLabeledImgs && arrayDispose(sLabeledImgs)
        setLabeledImgs(values)
    }

    const handleLabeledCapture = async (label: string): Promise<ILabeledImage | void> => {
        logger('handleLabeldCapture')
        if (webcamRef.current) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            const result = await webcamRef.current.capture()
            if (result) {
                genImgUid()
                const file: ILabeledImage = {
                    uid: sImgUid.toString(),
                    name: `${label}_${sImgUid.toString()}`,
                    tensor: result
                }
                return file
            }
        }
    }

    /***********************
     * Render
     ***********************/

    const _tensorX = sTrainSet?.xs as tf.Tensor4D
    const _tensorY = sTrainSet?.ys as tf.Tensor

    return (
        <Row gutter={16}>
            <h1>Posenet</h1>
            <Col span={12}>
                <Card title='Prediction' size='small'>
                    <WebCamera ref={webcamRef} model={sModel} onSubmit={handlePredict} prediction={sPredictResult}
                        labelsMap={sLabelsMap} isPreview />
                </Card>
                <Card>
                    <LabeledCaptureInputSet model={sModel} onSave={handleLabeledImagesSubmit}
                        onCapture={handleLabeledCapture}/>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Model(Expand from Mobilenet)' style={{ margin: '8px' }} size='small'>
                    <div>
                        <Button onClick={handleTrain} type='primary' style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                        <Button onClick={handleLoadModel} style={{ width: '30%', margin: '0 10%' }}> Load
                            Model </Button>
                        <div>status: {sStatus}</div>
                        <LabeledCaptureSetWidget model={sModel} labeledImgs={sLabeledImgs} onJsonLoad={handleLoadJson}/>
                        <div> XShape: {_tensorX?.shape.join(',')}, YShape: {_tensorY?.shape.join(',')}</div>
                    </div>
                    <div>
                        <TfvisModelWidget model={sModel}/>
                        <p>status: {sStatus}</p>
                    </div>
                    <div>
                        Select Layer : <Select onChange={handleLayerChange} defaultValue={0}>
                            {sLayersOption?.map((v) => {
                                return <Option key={v.index} value={v.index}>{v.name}</Option>
                            })}
                        </Select>
                        <TfvisLayerWidget layer={sCurLayer}/>
                    </div>

                    <p>backend: {sTfBackend}</p>
                </Card>
            </Col>
        </Row>
    )
}

export default MobilenetTransferWidget
