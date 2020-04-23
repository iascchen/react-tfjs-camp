import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Table } from 'antd'
import { formatTensorToStringArray, logger } from '../../utils'

const MAX_SAMPLES_COUNT = 20
const MAX_ARRAY_ITEM_DISPLAY = 100

interface IProps {
    xDataset: tf.Tensor
    yDataset: tf.Tensor
    pDataset?: tf.Tensor
    sampleCount?: number

    pageSize?: number
    debug?: boolean
}

const SentimentSampleDataVis = (props: IProps): JSX.Element => {
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
    }, [props.xDataset, sampleCount])

    useEffect(() => {
        if (!props.yDataset) {
            return
        }
        logger('init y')

        const _sampleInfo = props.yDataset.split(props.yDataset.shape[0])
        const _data = _sampleInfo.slice(0, sampleCount)
        setYData(_data)
    }, [props.yDataset, sampleCount])

    useEffect(() => {
        if (!props.pDataset) {
            return
        }
        logger('init p')

        const _sampleInfo = props.pDataset.split(props.pDataset.shape[0])
        const _data = _sampleInfo.slice(0, sampleCount)
        setPData(_data)
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
                    const yArray = formatTensorToStringArray(record.y as tf.Tensor, 0)
                    const yStr = yArray.length > 1 ? `[${yArray.join(', ')}]` : yArray.join(', ')
                    const color = 'red'
                    return <span style={{ color: color }}>{yStr}</span>
                }
            },
            {
                title: 'P',
                dataIndex: 'p',
                key: 'p',
                render: (text: string, record: tf.TensorContainerObject): JSX.Element => {
                    const pArray = formatTensorToStringArray(record.p as tf.Tensor, 2)
                    const pStr = pArray.length > 1 ? `[${pArray.join(', ')}]` : pArray.join(', ')
                    const color = 'green'
                    return pStr ? <span style={{ color: color }}>{pStr}</span> : <></>
                }
            }]
        setColumns(_columns)
    }, [])

    /***********************
     * Functions
     ***********************/

    const formatX = (sampleInfo: tf.Tensor): string => {
        // logger(sampleInfo.shape)

        const _array = formatTensorToStringArray(sampleInfo)
        if (_array.length > MAX_ARRAY_ITEM_DISPLAY) {
            return _array.slice(0, MAX_ARRAY_ITEM_DISPLAY).join(', ') + '...'
        } else {
            return _array.join(', ')
        }
    }

    return <Table columns={columns} dataSource={data} pagination={{ pageSize: props.pageSize ?? MAX_SAMPLES_COUNT }}/>
}

export default SentimentSampleDataVis
