import React, { useCallback, useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Table } from 'antd'
import { arrayDispose, ISampleInfo, logger } from '../../../utils'
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

export const prepareSamples = (_tensor: tf.Tensor, maxRow: number = MAX_SAMPLES_COUNT): ISampleInfo => {
    // logger(_tensor.shape)
    const length = (maxRow < _tensor?.shape[0]) ? maxRow : _tensor?.shape[0]
    const shapeSize = _tensor?.size / _tensor?.shape[0]
    const shape = _tensor?.shape.slice(1)
    const shapeStr = shape?.join(', ') || '1'
    const data = Array.from(_tensor?.dataSync()).slice(0, length * shapeSize)

    return { data, shape, shapeStr, shapeSize, length }
}

const formatShape = (sampleInfo: ISampleInfo, index: number, fraction = 0): JSX.Element => {
    return (
        <>
            {sampleInfo?.data.slice(index * sampleInfo?.shapeSize, (index + 1) * sampleInfo?.shapeSize)
                .map((_item, _idx) => {
                    return _item.toFixed(fraction)
                }).join(' , ')
            }
        </>
    )
}

const formatImage = (sampleInfo: ISampleInfo, index: number): JSX.Element => {
    const data = sampleInfo ? sampleInfo.data.slice(index * sampleInfo?.shapeSize, (index + 1) * sampleInfo?.shapeSize) : []
    const shapeArg = sampleInfo?.shape.slice(0, 2) as [number, number]
    return <RowImageWidget data={data} shape={shapeArg}/>
}

const SampleDataVis = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sampleCount] = useState(props.sampleCount)
    const [acc, setAcc] = useState(0)
    const [xData, setXData] = useState<ISampleInfo>()
    const [yData, setYData] = useState<ISampleInfo>()
    const [pData, setPData] = useState<ISampleInfo>()
    const [data, setData] = useState()

    const [columns, setColumns] = useState()

    /***********************
     * useCallback
     ***********************/

    const formatX = useCallback((sampleInfo: ISampleInfo, index: number) => {
        return props.xIsImage
            ? formatImage(sampleInfo, index)
            : formatShape(sampleInfo, index, props.xFloatFixed)
    }, [props.xFloatFixed, props.xIsImage])

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        if (!props.xDataset) {
            return
        }
        logger('init x')

        const _sampleInfo = prepareSamples(props.xDataset, sampleCount)
        setXData(_sampleInfo)

        return () => {
            logger('Dispose x')
            arrayDispose(_sampleInfo.data)
        }
    }, [props.xDataset, sampleCount])

    useEffect(() => {
        if (!props.yDataset) {
            return
        }
        logger('init y')

        const _sampleInfo = prepareSamples(props.yDataset, sampleCount)
        setYData(_sampleInfo)

        return () => {
            logger('Dispose y')
            arrayDispose(_sampleInfo.data)
        }
    }, [props.yDataset, sampleCount])

    useEffect(() => {
        if (!props.pDataset) {
            return
        }
        logger('init p')

        const _sampleInfo = prepareSamples(props.pDataset, sampleCount)
        setPData(_sampleInfo)

        return () => {
            logger('Dispose p')
            arrayDispose(_sampleInfo.data)
        }
    }, [props.pDataset, sampleCount])

    useEffect(() => {
        if (!xData || !yData) {
            return
        }
        logger('init sample data [x,y] ...')

        const _data = yData.data.map((v: number, i: number) => {
            return { key: i, x: formatX(xData, i), y: v }
        })
        setData(_data)

        return () => {
            logger('Dispose sample data [x,y] ...')
            arrayDispose(_data)
        }
    }, [formatX, xData, yData])

    useEffect(() => {
        if (!xData || !yData || !pData) {
            return
        }
        logger('init sample data [p] ...')
        const _data = pData.data.map((v: number, i: number) => {
            return pData
                ? { key: i, x: formatX(xData, i), y: yData.data[i], p: v }
                : null
        })
        setData(_data)

        const correct = pData.data.reduce((p, c, i, _array): number => {
            return c === yData.data[i] ? p + 1 : p
        }, 0)
        setAcc(correct / pData.data.length)
        console.log('Acc = ', acc, pData.data.length)

        return () => {
            logger('Dispose sample data [p] ...')
            arrayDispose(_data)
        }
    }, [formatX, pData, xData, yData])

    useEffect(() => {
        const _columns = [
            {
                title: 'X',
                dataIndex: 'x',
                key: 'x'
            },
            {
                title: 'Y',
                dataIndex: 'y',
                key: 'y',
                render: (text: string, record: any) => {
                    const color = (record.y === record.p) ? 'green' : 'red'
                    return (
                        <span style={{ color: color }}>{text}</span>
                    )
                }
            },
            {
                title: 'P',
                dataIndex: 'p',
                key: 'p',
                render: (text: string, record: any) => {
                    const color = (record.y === record.p) ? 'green' : 'red'
                    return (
                        <span style={{ color: color }}>{text}</span>
                    )
                }
            }]
        setColumns(_columns)
    }, [])

    /***********************
     * Functions
     ***********************/

    return (
        <div>
            Acc = {acc}
            <Table columns={columns} dataSource={data} pagination={{ pageSize: props.pageSize }}/>
        </div>
    )
}

export default SampleDataVis
