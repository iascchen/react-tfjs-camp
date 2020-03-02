import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, message, Row, Select } from 'antd'

import {logger, STATUS} from '../../../utils'
import SampleDataVis from '../../common/tensor/SampleDataVis'
import { JenaWeatherData } from './dataJena'

interface IProps{
    numFeatures: number

    onChange?: (model: JenaWeatherData) => void
}

const JenaDataWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()
    const [sSampleData, setSampleData] = useState<tf.TensorContainerObject>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data ...')

        setStatus(STATUS.LOADING)
        const dataHandler = new JenaWeatherData()
        dataHandler.load().then(() => {
            setDataHandler(dataHandler)

            const _sample = dataHandler.getSampleData()
            setSampleData(_sample)

            setStatus(STATUS.LOADED)
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

    return (
        <Row>
            <Col span={12}>
                <Card title={'Data'} style={{ margin: 8 }}>
                    {/*{JSON.stringify(sSampleData)}*/}
                    {/*{sSampleData && (<SampleDataVis xFloatFixed={4} xDataset={sSampleData.data as tf.Tensor}*/}
                    {/*    yDataset={sSampleData.normalizedTimeOfDay as tf.Tensor} ></SampleDataVis>)}*/}

                    {sStatus} {sDataHandler?.getDataLength()}
                </Card>
            </Col>
            <Col span={24} >
                <h3>Please goto ./public/data folder and download data file with follow command</h3>
                <code>wget https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv</code>
            </Col>
        </Row>
    )
}

export default JenaDataWidget
