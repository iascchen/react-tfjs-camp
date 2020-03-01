import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, Row, Select } from 'antd'

import { ILayerSelectOption, logger } from '../../utils'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'

import { buildGRUModel, buildLinearRegressionModel, buildMLPModel, buildSimpleRNNModel } from './modelJena'

const { Option } = Select

const ModelOptions = ['linear-regression', 'mlp', 'mlp-l2', 'mlp-dropout', 'simpleRnn', 'gru']

interface IProps{
    numFeatures: number

    onChange?: (model: tf.LayersModel) => void
}

const JenaModelWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sModelName, setModelName] = useState('simpleRnn')
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        const lookBack = 10 * 24 * 6 // Look back 10 days.
        const step = 6 // 1-hour steps.
        const numTimeSteps = Math.floor(lookBack / step)
        const numFeatures = props.numFeatures ?? 10
        const inputShape: tf.Shape = [numTimeSteps, numFeatures]

        let _model: tf.LayersModel
        switch (sModelName) {
            case 'linear-regression' :
                _model = buildLinearRegressionModel(inputShape)
                break
            case 'mlp' :
                _model = buildMLPModel(inputShape)
                break
            case 'mlp-l2' :
                _model = buildMLPModel(inputShape, { kernelRegularizer: tf.regularizers.l2() })
                break
            case 'mlp-dropout' :
                _model = buildMLPModel(inputShape, { dropoutRate: 0.25 })
                break
            case 'gru' :
                _model = buildGRUModel(inputShape)
                break
            case 'simpleRnn' :
            default:
                _model = buildSimpleRNNModel(inputShape)
                break
        }
        // _model.summary()
        setModel(_model)

        const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(_layerOptions)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [sModelName, props.numFeatures])

    useEffect(() => {
        if (!sModel || !props.onChange) {
            return
        }
        props.onChange(sModel)
    }, [sModel, props.onChange])

    /***********************
     * Render
     ***********************/

    const handleModelChange = (value: string): void => {
        setModelName(value)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const modelTitle = (
        <>
            Model
            <Select onChange={handleModelChange} defaultValue={'simpleRnn'} style={{ marginLeft: 16 }}>
                {ModelOptions.map((v) => {
                    return <Option key={v} value={v}>{v}</Option>
                })}
            </Select>
        </>
    )

    const layerTitle = (
        <>
            {sModelName} Layers
            <Select onChange={handleLayerChange} defaultValue={0} style={{ marginLeft: 16 }}>
                {sLayersOption?.map((v) => {
                    return <Option key={v.index} value={v.index}>{v.name}</Option>
                })}
            </Select>
        </>
    )

    return (
        <Row>
            <Col span={12} >
                <Card title={modelTitle} style={{ margin: 8 }}>
                    { sModel && <TfvisModelWidget model={sModel}/>}
                </Card>
            </Col>
            <Col span={12}>
                <Card title={layerTitle} style={{ margin: 8 }}>
                    <TfvisLayerWidget layer={curLayer}/>
                </Card>
            </Col>
        </Row>
    )
}

export default JenaModelWidget
