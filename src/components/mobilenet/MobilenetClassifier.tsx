import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, Row, Select } from 'antd'

import { logger, STATUS } from '../../utils'
// import SampleDataVis from '../common/tensor/SampleDataVis'
// import TfvisHistoryWidget from '../common/tfvis/TfvisHistoryWidget'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import LabeledImageItem from '../common/tensor/LabeledImageItem'
import LabeledImageWidget from '../common/tensor/LabeledImageWidget'
// import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'

const { Option } = Select

const IMAGE_SIZE = 224
// const TOPK_PREDICTIONS = 10

// const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
const MOBILENET_MODEL_PATH = '/model/mobilenet_v1_0.25_224/model.json'

interface ILayerSelectOption {
    name: string
    index: number
}

const MobilenetClassifier = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)

    const [model, setModel] = useState<tf.LayersModel>()
    const [layersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [predictResult, setPredictResult] = useState<tf.Tensor>()

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
                _model = mobilenet

                // Warmup the model. This isn't necessary, but makes the first prediction
                // faster. Call `dispose` to release the WebGL memory allocated for the return
                // value of `predict`.
                const _temp = _model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])) as tf.Tensor
                _temp.dispose()

                setModel(_model)

                const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
                    return { name: l.name, index }
                })
                setLayersOption(_layerOptions)

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
     * useEffects only for dispose
     ***********************/

    /***********************
     * Functions
     ***********************/

    const handlePredict = (imageTensor: tf.Tensor): void => {
        if (!imageTensor) {
            return
        }
        const [p] = tf.tidy(() => {
            const _sample = tf.image.resizeBilinear(imageTensor as tf.Tensor4D, [224, 224])
            const offset = tf.scalar(127.5)
            // Normalize the image from [0, 255] to [-1, 1].
            const normalized = _sample.sub(offset).div(offset)
            // Reshape to a single-element batch so we can pass it to predict.
            const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3])

            const result = model?.predict(batched) as tf.Tensor
            console.log(result)

            const p = result?.argMax(-1)
            return [p]
        })

        setPredictResult(p)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = model?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    // const handleLabeledImageItemChange = (value: string): void => {
    //     const obj = JSON.parse(value)
    //     logger(obj)
    // }

    const handleLabeledImagesSubmit = (value: object): void => {
        console.log('handleLabeledImagesSubmit', value)
        // try {
        //     await this.props.save(recipeBody)
        // } catch (error) {
        //     if (error.errorCode && error.errorCode === API_SERVER_ERROR.INPUT_ERROR) {
        //         return Form.inputServerError(error, this.formRefs)
        //     }
        //     throw error
        // }
        // this.setState({ loading: false })
        // const messageAndUrl = getMessageAndUrl(this.props.id, false, false, this.props.intl)
        // AlertSuccess(messageAndUrl.message, { position: messageAndUrl.position })
        // this.props.push(messageAndUrl.url)
        //
        // if (!this.props.id) {
        //     this.props.push(recipe.fill())
        // } else {
        //     await this.props.getRecipe()
        // }
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row gutter={16}>
            <h1>MNIST</h1>

            <Col span={12}>
                <Card title='Predict' style={{ margin: '8px' }} size='small'>
                    <ImageUploadWidget model={model} onSubmit={handlePredict} prediction={predictResult}/>
                    {/* <TfvisHistoryWidget logMsg={logMsg} debug /> */}
                </Card>
                <Card title='Basic Model' style={{ margin: '8px' }} size='small'>
                    <div>
                        <TfvisModelWidget model={model}/>
                        <p>status: {status}</p>
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
                <Card title='Machine Learning(KNN)' style={{ margin: '8px' }} size='small'>
                    <div>Labeled Images</div>
                    <LabeledImageWidget model={model} labeledImageList={[]}
                        onSubmit={handleLabeledImagesSubmit} />

                    {/* <LabeledImageItem index={0} onChange={handleLabeledImageItemChange}></LabeledImageItem> */}
                    {/* <SampleDataVis xDataset={predictSet?.xs as tf.Tensor} yDataset={predictSet?.ys as tf.Tensor} */}
                    {/*    pDataset={predictResult} xIsImage /> */}
                </Card>
            </Col>
        </Row>
    )
}

export default MobilenetClassifier
