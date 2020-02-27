import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row } from 'antd'

import {
    ILabeledImageSet,
    logger,
    STATUS
} from '../../utils'
import { MOBILENET_IMAGE_SIZE, MOBILENET_MODEL_PATH } from '../../constant'
import WebCamera from '../common/tensor/WebCamera'

const MobilenetTransferWidget = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<tf.LayersModel>()

    // const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.LOADING)

        let _model: tf.LayersModel
        tf.loadLayersModel(MOBILENET_MODEL_PATH).then(
            (mobilenet) => {
                // Constact posenet

                setModel(mobilenet)
                setStatus(STATUS.LOADED)
            },
            (error) => {
                logger(error)
            }
        )

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [])

    /***********************
     * Functions
     ***********************/

    // const formatImageForMobilenet = (imgTensor: tf.Tensor, imageSize: number): tf.Tensor => {
    //     const sample = tf.image.resizeBilinear(imgTensor as tf.Tensor3D, [imageSize, imageSize])
    //     logger(JSON.stringify(sample))
    //
    //     const offset = tf.scalar(127.5)
    //     // Normalize the image from [0, 255] to [-1, 1].
    //     const normalized = sample.sub(offset).div(offset)
    //     // Reshape to a single-element batch so we can pass it to predict.
    //     return normalized.reshape([1, imageSize, imageSize, 3])
    // }

    const train = (imageSetList: ILabeledImageSet[]): void => {
        logger('train', imageSetList)

        setStatus(STATUS.TRAINING)
        // const training = new Promise<void>(async (resolve, reject) => {
        //     for (const imgSet of imageSetList) {
        //         const { label, imageList } = imgSet
        //         if (imageList) {
        //             for (const imgItem of imageList) {
        //                 const imgBase64 = imgItem.img
        //                 if (imgBase64) {
        //                     const _imgData = await getImageDataFromBase64(imgBase64)
        //                     const _imgTensor = tf.browser.fromPixels(_imgData, 3)
        //                     // const _imgBatched = formatImageForMobilenet(_imgTensor, MOBILENET_IMAGE_SIZE)
        //                     const _imgFeature = sModel?.predict(_imgTensor) as tf.Tensor
        //                 }
        //             }
        //         }
        //     }
        //     resolve()
        // })
        //
        // training.then(
        //     () => {
        //         setStatus(STATUS.TRAINED)
        //     }
        // )
    }

    const handleTrain = (): void => {
        // if (sLabeledImgs) {
        //     train(sLabeledImgs)
        // }
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

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>Posenet</h1>
            <Col span={12}>
                <Card title='Prediction' style={{ margin: '8px' }} size='small'>
                    <WebCamera model={sModel} onSubmit={handlePredict} prediction={sPredictResult}/>
                </Card>
            </Col>
            <Col span={12}>
                <Card title='Model(Expand from Mobilenet)' style={{ margin: '8px' }} size='small'>
                    <div>
                        <Button onClick={handleLoadWeight} type='primary' style={{ width: '30%', margin: '0 10%' }}> Load Weights </Button>
                        <div>status: {sStatus}</div>

                    </div>
                    <p>backend: {sTfBackend}</p>
                </Card>
            </Col>
        </Row>
    )
}

export default MobilenetTransferWidget
