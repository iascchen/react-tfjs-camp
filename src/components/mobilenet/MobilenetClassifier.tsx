import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, Row, Select } from 'antd'

import { logger, STATUS, ILayerSelectOption } from '../../utils'
import { MOBILENET_IMAGE_SIZE, MOBILENET_MODEL_PATH } from '../../constant'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'

const { Option } = Select

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
                const _temp = _model.predict(tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])) as tf.Tensor
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
            const batched = normalized.reshape([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])

            const result = model?.predict(batched) as tf.Tensor
            logger(result)

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

    /***********************
     * Render
     ***********************/

    return (
        <>
            <h1>Mobilenet Classifier</h1>
            <Row gutter={16}>
                <Col span={12}>
                    <Card title='Predict' style={{ margin: '8px' }} size='small'>
                        <ImageUploadWidget model={model} onSubmit={handlePredict} prediction={predictResult}/>
                    </Card>
                    <Card title='Infomation' style={{ margin: '8px' }} size='small'>
                        <p>此处显示说明 MD 文件</p>
                    </Card>
                </Col>
                <Col span={12}>
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
            </Row>
        </>
    )
}

export default MobilenetClassifier
