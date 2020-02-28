import React, {useEffect, useReducer, useRef, useState} from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, message, Row, Select } from 'antd'

import {
    arrayDispose,
    ILabeledImage,
    ILabeledImageFileJson,
    ILabeledImageSet,
    ILayerSelectOption,
    logger,
    STATUS
} from '../../utils'
import { MOBILENET_IMAGE_SIZE } from '../../constant'
import WebCamera, { IWebCameraHandler } from '../common/tensor/WebCamera'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import { createModel } from './modelTransfer'
import { TransferDataset } from './dataTransfer'
import LabeledCaptureInputSet from '../common/tensor/LabeledCaptureInputSet'
import LabeledImageSetWidget from '../common/tensor/LabeledImageSetWidget'

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
    const [sLearningRate, setLearningRate] = useState<number>(0.15)
    const [sDenseUnits, setDenseUnits] = useState<number>(10)

    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sBatchSize, setBatchSize] = useState<number>(10)
    const [sEpochs, setEpochs] = useState<number>(10)
    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()

    const [sImgUid, genImgUid] = useReducer((x: number) => x + 1, 0)

    // const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    const webcamRef = useRef<IWebCameraHandler>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.LOADING)

        let _model: tf.LayersModel
        createModel(sOutputClasses, sLearningRate, sDenseUnits).then(
            (result) => {
                _model = result
                setModel(_model)
                setStatus(STATUS.LOADED)

                const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
                    return { name: l.name, index }
                })
                setLayersOption(_layerOptions)
            },
            (e) => {
                logger(e)
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                message.error(e.message)
            }
        )

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [sOutputClasses, sLearningRate, sDenseUnits])

    useEffect(() => {
        logger('init data set ...')

        const tSet = new TransferDataset(sOutputClasses)
        setTrainSet(() => tSet.getData()) // when use sTrainSet, will get last records

        return () => {
            logger('Train Data Dispose')
            tSet.dispose()
        }
    }, [sOutputClasses])

    /***********************
     * Functions
     ***********************/

    const train = (_trainSet: tf.TensorContainerObject): void => {
        logger('train', _trainSet)

        setStatus(STATUS.TRAINING)

        // // We parameterize batch size as a fraction of the entire dataset because the
        // // number of examples that are collected depends on how many examples the user
        // // collects. This allows us to have a flexible batch size.
        // const batchSize =
        //     Math.floor(_trainSet.xs.shape[0] * sBatchSize)
        // if (!(batchSize > 0)) {
        //     throw new Error('Batch size is 0 or NaN. Please choose a non-zero fraction.')
        // }
        //
        // const surface = { name: 'Logs', tab: 'Train Logs' }
        // // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
        // sModel?.fit(_trainSet.xs, _trainSet.ys, {
        //     batchSize,
        //     epochs: sEpochs,
        //     callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc', 'val_loss', 'val_acc'])
        // })
    }

    const handleTrain = (): void => {
        sTrainSet && train(sTrainSet)
    }

    const predict = async (): Promise<void> => {
        setStatus(STATUS.PREDICTING)

        while (sStatus === STATUS.PREDICTING) {
            // // Capture the frame from the webcam.
            // const img = await getImage()
            //
            // // Make a prediction through mobilenet, getting the internal activation of
            // // the mobilenet model, i.e., "embeddings" of the input images.
            // const embeddings = truncatedMobileNet.predict(img)
            //
            // // Make a prediction through our newly-trained model using the embeddings
            // // from mobilenet as input.
            // const predictions = sModel.predict(embeddings)
            //
            // // Returns the index with the maximum probability. This number corresponds
            // // to the class the model thinks is the most probable given the input.
            // const predictedClass = predictions.as1D().argMax()
            // const classId = (await predictedClass.data())[0]
            // img.dispose()
            //
            // ui.predictClass(classId)
            await tf.nextFrame()
        }
        setStatus(STATUS.PREDICTED)
    }

    const handlePredict = (imgTensor: tf.Tensor): void => {
        if (!imgTensor) {
            return
        }
        // console.log('handlePredict', imgTensor)
        const [imgFeature] = tf.tidy(() => {
            // const batched = formatImageForMobilenet(imgTensor, MOBILENET_IMAGE_SIZE)
            const batched = imgTensor.reshape([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])
            const result = sModel?.predict(batched) as tf.Tensor
            // logger('handlePredict result ', result)

            const imgFeature = result?.argMax(-1)
            return [imgFeature]
        })
        setPredictResult(imgFeature)
    }

    const handleLoadWeight = (): void => {
        // setLabeledImgs(values)
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
                    img: result
                }
                return file
            }
        }
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>Posenet</h1>
            <Col span={12}>
                <Card title='Prediction' size='small'>
                    <WebCamera model={sModel} onSubmit={handlePredict} prediction={sPredictResult}
                        ref={webcamRef} isPreview />
                </Card>
                <Card>
                    <LabeledCaptureInputSet model={sModel} onSave={handleLabeledImagesSubmit}
                        onCapture={handleLabeledCapture}/>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Model(Expand from Mobilenet)' style={{ margin: '8px' }} size='small'>
                    <div>
                        <Button onClick={handleLoadWeight} type='primary' style={{ width: '30%', margin: '0 10%' }}> Load
                            Weights </Button>
                        <div>status: {sStatus}</div>
                        <LabeledImageSetWidget model={sModel} labeledImgs={sLabeledImgs} onJsonLoad={handleLoadJson}/>
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
