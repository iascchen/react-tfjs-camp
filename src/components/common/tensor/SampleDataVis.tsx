import React, { useCallback, useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Table } from 'antd'
import { arrayDispose, formatTensorToStringArray, getTensorLabel, logger } from '../../../utils'
import RowImageWidget from './RowImageWidget'

const MAX_SAMPLES_COUNT = 20

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

const formatImage = (sampleInfo: tf.Tensor): JSX.Element => {
    const data = Array.from(sampleInfo.dataSync())
    const shapeArg = sampleInfo.shape.slice(1, 3) as [number, number]
    return <RowImageWidget data={data} shape={shapeArg}/>
}

const SampleDataVis = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sampleCount] = useState(props.sampleCount)
    const [acc, setAcc] = useState(0)

    const [xData, setXData] = useState<tf.Tensor[]>()
    const [yData, setYData] = useState<tf.Tensor[]>()
    const [pData, setPData] = useState<tf.Tensor[]>()
    const [yDataLabel, setYDataLabel] = useState<string[]>([])
    const [pDataLabel, setPDataLabel] = useState<string[]>([])

    const [data, setData] = useState()

    const [columns, setColumns] = useState()

    /***********************
     * useCallback
     ***********************/

    const formatX = useCallback((sampleInfo: tf.Tensor) => {
        return props.xIsImage
            ? formatImage(sampleInfo)
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
        const _sampleLabel = getTensorLabel(_data)
        setYData(_data)
        setYDataLabel(_sampleLabel)

        return () => {
            logger('Dispose y')
            arrayDispose(_sampleInfo)
            arrayDispose(_sampleLabel)
        }
    }, [props.yDataset, sampleCount])

    useEffect(() => {
        if (!props.pDataset) {
            return
        }
        logger('init p')

        const _sampleInfo = props.pDataset.split(props.pDataset.shape[0])
        const _data = _sampleInfo.slice(0, sampleCount)
        const _sampleLabel = getTensorLabel(_data)
        setPData(_data)
        setPDataLabel(_sampleLabel)

        return () => {
            logger('Dispose p')
            arrayDispose(_sampleInfo)
            arrayDispose(_sampleLabel)
        }
    }, [props.pDataset, sampleCount])

    useEffect(() => {
        if (!xData || !yData) {
            return
        }
        logger('init sample data [x,y] ...')

        const _data = yData.map((v, i) => {
            return { key: i, x: xData[i], y: v, yLabel: yDataLabel[i] }
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
                ? { key: i, x: xData[i], y: yData[i], p: v, yLabel: yDataLabel[i], pLabel: pDataLabel[i] }
                : null
        })
        setData(_data)

        const correct = pData.reduce((p, c, i, _array): number => {
            return pDataLabel[i] === yDataLabel[i] ? p + 1 : p
        }, 0)
        setAcc(correct / pData.length)
        logger('Acc = ', acc, pData.length)

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
                    const yArray = formatTensorToStringArray(record.y as tf.Tensor, 0)
                    const yStr = yArray.length > 1 ? `[${yArray.join(', ')}] => ${record.yLabel}` : yArray.join(', ')
                    const color = record.yLabel === record.pLabel ? 'green' : 'red'
                    return <span style={{ color: color }}>{yStr}</span>
                }
            },
            {
                title: 'P',
                dataIndex: 'p',
                key: 'p',
                render: (text: string, record: tf.TensorContainerObject): JSX.Element => {
                    const pArray = formatTensorToStringArray(record.p as tf.Tensor, 2)
                    const pStr = pArray.length > 1 ? `[${pArray.join(', ')}] => ${record.pLabel}` : pArray.join(', ')
                    const color = record.yLabel === record.pLabel ? 'green' : 'red'
                    return pStr ? <span style={{ color: color }}>{pStr}</span> : <></>
                }
            }]
        setColumns(_columns)
    }, [])

    /***********************
     * Functions
     ***********************/

    return (
        <div>
            {pData && <span>Accuracy = {(acc * 100).toFixed(0) + '%'}</span>}
            <Table columns={columns} dataSource={data} pagination={{ pageSize: props.pageSize ?? MAX_SAMPLES_COUNT }}/>
        </div>
    )
}

export default SampleDataVis
