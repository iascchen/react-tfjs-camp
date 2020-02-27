import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as knnClassifier from '@tensorflow-models/knn-classifier'
import { Button, Card, Col, message, Row } from 'antd'

import {
    getImageDataFromBase64,
    IKnnPredictResult,
    ILabeledImageFileJson,
    ILabeledImageSet,
    logger,
    STATUS
} from '../../utils'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import LabeledImageInputSet from '../common/tensor/LabeledImageInputSet'
import LabeledImageSetWidget from '../common/tensor/LabeledImageSetWidget'

const MOBILENET_IMAGE_SIZE = 224
const KNN_TOPK = 10

// const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
const MOBILENET_MODEL_PATH = '/model/mobilenet_v1_0.25_224/model.json'

const MobilenetClassifier = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sKnn, setKnn] = useState<knnClassifier.KNNClassifier>()
    const [sModel, setModel] = useState<tf.LayersModel>()

    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor | IKnnPredictResult >()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.LOADING)

        const _knn = knnClassifier.create()
        setKnn(_knn)

        let _model: tf.LayersModel
        tf.loadLayersModel(MOBILENET_MODEL_PATH).then(
            (mobilenet) => {
                // Return a model that outputs an internal activation.
                const layer = mobilenet.getLayer('conv_preds')
                _model = tf.model({ inputs: mobilenet.inputs, outputs: layer.output })

                // Warmup the model. This isn't necessary, but makes the first prediction
                // faster. Call `dispose` to release the WebGL memory allocated for the return
                // value of `predict`.
                const _temp = _model.predict(tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])) as tf.Tensor
                _temp.dispose()

                setModel(_model)
                setStatus(STATUS.LOADED)
            },
            (error) => {
                logger(error)
            }
        )

        return () => {
            logger('Model Dispose')
            _model?.dispose()
            _knn.dispose()
        }
    }, [])

    /***********************
     * Functions
     ***********************/

    const formatImageForMobilenet = (imgTensor: tf.Tensor, imageSize: number): tf.Tensor => {
        const sample = tf.image.resizeBilinear(imgTensor as tf.Tensor4D, [imageSize, imageSize])

        const offset = tf.scalar(127.5)
        // Normalize the image from [0, 255] to [-1, 1].
        const normalized = sample.sub(offset).div(offset)
        // Reshape to a single-element batch so we can pass it to predict.
        return normalized.reshape([1, imageSize, imageSize, 3])
    }

    const train = (imageSetList: ILabeledImageSet[]): void => {
        logger('train', imageSetList)

        setStatus(STATUS.TRAINING)

        imageSetList?.forEach(imgSet => {
            const { label, imageList } = imgSet
            imageList?.forEach(imgItem => {
                const imgBase64 = imgItem.img
                if (imgBase64) {
                    getImageDataFromBase64(imgBase64).then((_imgData) => {
                        const _imgTensor = tf.browser.fromPixels(_imgData, 4)
                        const _imgBatched = formatImageForMobilenet(_imgTensor, MOBILENET_IMAGE_SIZE)
                        const _imgFeature = sModel?.predict(_imgBatched) as tf.Tensor

                        logger('sKnn.addExample', label, _imgFeature)
                        sKnn?.addExample(_imgFeature, label)
                    }, (error) => {
                        logger(error)
                    })
                }
            })

            setStatus(STATUS.TRAINED)
        })
    }

    const resetKnn = (): void => {
        sKnn?.clearAllClasses()
    }

    const handleTrain = (): void => {
        if (sLabeledImgs) {
            train(sLabeledImgs)
        }
    }

    const handleKnnReset = (): void => {
        resetKnn()
    }

    const handlePredict = async (imgTensor: tf.Tensor): Promise<void> => {
        if (!imgTensor) {
            return
        }
        const [imgFeature] = tf.tidy(() => {
            const batched = formatImageForMobilenet(imgTensor, MOBILENET_IMAGE_SIZE)
            const imgFeature = sModel?.predict(batched) as tf.Tensor
            // logger(imgFeature)
            return [imgFeature]
        })
        try {
            const res = await sKnn?.predictClass(imgFeature, KNN_TOPK)
            // logger('handlePredict', res)
            setPredictResult(res)
        } catch (e) {
            // logger(e)
            await message.error(e.message)
        }
    }

    const handleLabeledImagesSubmit = (value: ILabeledImageFileJson): void => {
        console.log('handleLabeledImagesSubmit', value)

        const labeledImageSetList = value.labeledImageSetList
        setLabeledImgs(labeledImageSetList)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>Mobilenet + KNN</h1>
            <Col span={12}>
                <Card title='Machine Learning(KNN)' style={{ margin: '8px' }} size='small'>
                    <div>Labeled Images</div>
                    <LabeledImageInputSet model={sModel} onSave={handleLabeledImagesSubmit} />
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Predict' style={{ margin: '8px' }} size='small'>
                    <ImageUploadWidget model={sModel} onSubmit={handlePredict} prediction={sPredictResult}/>
                </Card>
                <Card title='Mobilenet + KNN' style={{ margin: '8px' }} size='small'>
                    <div>
                        {/* <TfvisModelWidget model={model}/> */}

                        <Button onClick={handleTrain} type='primary'> Train </Button>
                        <Button onClick={handleKnnReset} > Reset Model </Button>
                        <p>status: {sStatus}</p>
                        <p>KNN: {sKnn?.getNumClasses()}</p>

                        <p>labeled Images : </p>
                        <LabeledImageSetWidget model={sModel} labeledImgs={sLabeledImgs} />
                    </div>
                    <p>backend: {sTfBackend}</p>
                </Card>
            </Col>
        </Row>
    )
}

export default MobilenetClassifier
