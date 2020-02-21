import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card } from 'antd'
import { Axis, Chart, Geom, Legend, Tooltip } from 'bizcharts'

import { arrayDispose, logger } from '../../utils'

const MAX_SAMPLES_COUNT = 100

interface IProps {
    xDataset?: tf.Tensor
    yDataset?: tf.Tensor
    pDataset?: tf.Tensor
    sampleCount?: number

    debug?: boolean
}

const prepareData = (_tensor: tf.Tensor, _sampleCount: number = MAX_SAMPLES_COUNT): number[] => {
    if (!_tensor) {
        return []
    }
    const _array = _tensor.dataSync()
    return Array.from(_array ?? []).splice(0, _sampleCount)
}

const CurveVis = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [xData, setXData] = useState()
    const [yData, setYData] = useState()
    const [pData, setPData] = useState()
    const [data, setData] = useState()
    const [sampleCount] = useState(props.sampleCount)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        if (!props.xDataset) {
            return
        }
        logger('init xDataset ...')

        const _d = prepareData(props.xDataset, sampleCount)
        setXData(_d)

        return () => {
            logger('Dispose xDataset ...')
            arrayDispose(_d)
        }
    }, [props.xDataset, sampleCount])

    useEffect(() => {
        if (!props.yDataset) {
            return
        }
        logger('init yDataset ...')

        const _d = prepareData(props.yDataset, sampleCount)
        setYData(_d)

        return () => {
            logger('Dispose yDataset ...')
            arrayDispose(_d)
        }
    }, [props.yDataset, sampleCount])

    useEffect(() => {
        if (!props.pDataset) {
            return
        }
        logger('init pDataset ...')

        const _d = prepareData(props.pDataset, sampleCount)
        setPData(_d)

        return () => {
            logger('Dispose pDataset ...')
            arrayDispose(_d)
        }
    }, [props.pDataset, sampleCount])

    useEffect(() => {
        logger('init sample data [x,y] ...')
        const _data = xData?.map((v: number, i: number) => {
            return { x: v, type: 'y', y: yData[i] }
        })
        setData(_data)

        return () => {
            logger('Dispose sample data [x,y] ...')
            arrayDispose(_data)
        }
    }, [xData, yData])

    useEffect(() => {
        logger('init sample data [p] ...')

        const _data: any[] = []
        pData?.forEach((v: number, i: number) => {
            _data.push({ x: xData[i], type: 'p', y: v })
            _data.push({ x: xData[i], type: 'y', y: yData[i] })
        })
        setData(_data)

        return () => {
            logger('Dispose sample data [p] ...')
            arrayDispose(_data)
        }
    }, [pData])

    /***********************
     * Functions
     ***********************/

    return (
        <Card>
            <Chart height={400} data={data} padding='auto' forceFit>
                <Axis name='X' />
                <Axis name='Y' />
                <Legend />
                <Tooltip/>
                <Geom type='line' position='x*y' size={2} color={'type'} shape={'smooth'}/>
            </Chart>
            Sample count : { props.sampleCount }
            {props.debug ? JSON.stringify(data) : ''}
        </Card>
    )
}

export default CurveVis
