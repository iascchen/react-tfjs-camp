import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, message, Row, Select } from 'antd'

import { ILayerSelectOption, logger } from '../../../utils'

import { JenaWeatherData } from './dataJena'

const { Option } = Select

interface IProps{
    numFeatures: number

    onChange?: (model: tf.data.Dataset<any>) => void
}

const JenaDataWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sNumFeatures, setNumFeatures] = useState(10)
    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data ...')

        const dataHandler = new JenaWeatherData()
        dataHandler.load().then(() => {
            setDataHandler(dataHandler)
        }, (e) => {
            logger(e)
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.error(e.message)
        })

        return () => {
            logger('Model Dispose')
            dataHandler.dispose()
        }
    }, [])

    useEffect(() => {
        if (!sDataHandler || !props.onChange) {
            return
        }
        props.onChange(sDataHandler)
    }, [sDataHandler, props.onChange])

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

                </Card>
            </Col>
            <Col span={12}>
                <Card title={layerTitle} style={{ margin: 8 }}>

                </Card>
            </Col>
        </Row>
    )
}

export default JenaDataWidget
