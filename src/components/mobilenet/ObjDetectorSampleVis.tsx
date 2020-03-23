import React, { useCallback, useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Table } from 'antd'

import { arrayDispose, formatTensorToStringArray, logger } from '../../utils'
import TensorImageThumbWidget from '../common/tensor/TensorImageThumbWidget'

const MAX_SAMPLES_COUNT = 20
const IMAGE_HEIGHT = 86

interface IProps {
    xDataset: tf.Tensor
    yDataset: tf.Tensor
    pDataset?: tf.Tensor
    sampleCount?: number

    xFloatFixed?: number
    xIsImage?: boolean
    pageSize?: number

    debug?: boolean
}

const formatImage = (sampleInfo: tf.Tensor3D): JSX.Element => {
    return <TensorImageThumbWidget data={sampleInfo} height={IMAGE_HEIGHT}/>
}

const ObjDetectorSampleVis = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sampleCount] = useState(props.sampleCount)

    const [xData, setXData] = useState<tf.Tensor[]>()
    const [yData, setYData] = useState<tf.Tensor[]>()
    const [pData, setPData] = useState<tf.Tensor[]>()

    const [data, setData] = useState()
    const [columns, setColumns] = useState()

    /***********************
     * useCallback
     ***********************/

    const formatX = useCallback((sampleInfo: tf.Tensor) => {
        return props.xIsImage
            ? formatImage(sampleInfo.squeeze())
            : formatTensorToStringArray(sampleInfo, props?.xFloatFixed).join(', ')
    }, [props.xFloatFixed, props.xIsImage])

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        if (!props.xDataset) {
            return
        }
        logger('init x')

        const _sampleInfo = props.xDataset.split(props.xDataset.shape[0])
        const _data = _sampleInfo.slice(0, sampleCount)
        setXData(_data)

        return () => {
            logger('Dispose x')
            arrayDispose(_sampleInfo)
        }
    }, [props.xDataset, sampleCount])

    useEffect(() => {
        if (!props.yDataset) {
            return
        }
        logger('init y')

        const _sampleInfo = props.yDataset.split(props.yDataset.shape[0])
        const _data = _sampleInfo.slice(0, sampleCount)
        setYData(_data)

        return () => {
            logger('Dispose y')
            arrayDispose(_sampleInfo)
            // arrayDispose(_sampleLabel)
        }
    }, [props.yDataset, sampleCount])

    useEffect(() => {
        if (!props.pDataset) {
            return
        }
        logger('init p')

        const _sampleInfo = props.pDataset.split(props.pDataset.shape[0])
        const _data = _sampleInfo.slice(0, sampleCount)
        setPData(_data)

        return () => {
            logger('Dispose p')
            arrayDispose(_sampleInfo)
        }
    }, [props.pDataset, sampleCount])

    useEffect(() => {
        if (!xData || !yData) {
            return
        }
        logger('init sample data [x,y] ...')

        const _data = yData.map((v, i) => {
            return { key: i, x: xData[i], y: v }
        })
        setData(_data)

        return () => {
            logger('Dispose sample data [x,y] ...')
            arrayDispose(_data)
        }
    }, [xData, yData])

    useEffect(() => {
        if (!xData || !yData || !pData) {
            return
        }
        logger('init sample data [p] ...')
        const _data = pData.map((v, i) => {
            return pData
                ? { key: i, x: xData[i], y: yData[i], p: v }
                : null
        })
        setData(_data)

        return () => {
            logger('Dispose sample data [p] ...')
            arrayDispose(_data)
        }
    }, [pData, xData, yData])

    useEffect(() => {
        const _columns = [
            {
                title: 'X',
                dataIndex: 'x',
                key: 'x',
                render: (text: string, record: tf.TensorContainerObject): JSX.Element => {
                    return <span>{formatX(record.x as tf.Tensor)}</span>
                }
            },
            {
                title: 'Y',
                dataIndex: 'y',
                key: 'y',
                render: (text: string, record: tf.TensorContainerObject): JSX.Element => {
                    const yArray = formatTensorToStringArray(record.y as tf.Tensor, 3)
                    const yStr = yArray.join(', ')
                    return <span>{yStr}</span>
                }
            },
            {
                title: 'P',
                dataIndex: 'p',
                key: 'p',
                render: (text: string, record: tf.TensorContainerObject): JSX.Element => {
                    const pArray = formatTensorToStringArray(record.p as tf.Tensor, 3)
                    const pStr = pArray.join(', ')
                    return pStr ? <span>{pStr}</span> : <></>
                }
            }]
        setColumns(_columns)
    }, [])

    /***********************
     * Functions
     ***********************/

    return (
        <div>
            <Table columns={columns} dataSource={data} pagination={{ pageSize: props.pageSize ?? MAX_SAMPLES_COUNT }}/>
        </div>
    )
}

export default ObjDetectorSampleVis
