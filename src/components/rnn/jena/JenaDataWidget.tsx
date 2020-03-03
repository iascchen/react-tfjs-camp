import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row, message } from 'antd'

import { logger, STATUS } from '../../../utils'
import MarkdownWidget from '../../common/MarkdownWidget'

import { JenaWeatherData } from './dataJena'

interface IProps {
    onChange?: (model: JenaWeatherData) => void
}

const mdInfo = '**注意** \n' +
    '\n' +
    '* 如果您要在本地环境运行这个例子，最好预先下载数据文件。并将数据文件放在此项目的 `./public/data` 目录下。\n' +
    '\n' +
    '    [https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv](https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv)\n' +
    '\n' +
    '* 所需的数据大约有 41.2MB。\n' +
    '* 刷新页面，会丢失已经加载的数据。'

const JenaDataWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)
    const [sLoadSpendSec, setLoadSpendSec] = useState<number>(0)

    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()
    const [sSampleData, setSampleData] = useState<tf.TensorContainerObject>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data ...')
        const dataHandler = new JenaWeatherData()

        if (dataHandler.csvLines.length === 0) {
            dataHandler.loadCsv().then(
                (result) => {
                    dataHandler.csvLines = result
                    dataHandler.loadDataColumnNames()
                    logger('Data fetched', dataHandler.csvLines.length)
                    logger('dataColumnNames', dataHandler.dataColumnNames)
                }, (e) => {
                    logger(e)
                    // eslint-disable-next-line @typescript-eslint/no-floating-promises
                    message.error(e.message)
                })
        }
        setDataHandler(dataHandler)

        return () => {
            logger('Data Dispose')
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

    const handleDataLoad = (): void => {
        setStatus(STATUS.LOADING)

        const beginMs = performance.now()
        sDataHandler?.load().then(() => {
            const _sample = sDataHandler.getSampleData()
            setSampleData(_sample)

            setLoadSpendSec((performance.now() - beginMs) / 1000)
            setStatus(STATUS.LOADED)
        }, (e) => {
            logger(e)
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.error(e.message)
        })
    }

    const statusInfo = (): string => {
        if (sStatus === STATUS.LOADING) {
            return 'Please waiting...'
        } else if (sStatus === STATUS.LOADED) {
            return `${sDataHandler?.getDataLength()} items with ${sLoadSpendSec.toFixed(3)} s`
        } else {
            return 'No data load'
        }
    }

    return (
        <Row>
            <Col span={12}>
                <Card title={'Data'} style={{ margin: 8 }}>
                    <MarkdownWidget source={mdInfo}/>
                    <Button style={{ width: '30%', margin: '0 auto' }} type='primary'
                        disabled={sStatus === STATUS.LOADING} onClick={handleDataLoad}> Load </Button>
                    {/* {JSON.stringify(sSampleData)} */}
                    {/* {sSampleData && (<SampleDataVis xFloatFixed={4} xDataset={sSampleData.data as tf.Tensor} */}
                    {/*    yDataset={sSampleData.normalizedTimeOfDay as tf.Tensor} ></SampleDataVis>)} */}

                    <p>Status: {sStatus}</p>
                    <p>{statusInfo()}</p>
                </Card>
            </Col>
        </Row>
    )
}

export default JenaDataWidget
