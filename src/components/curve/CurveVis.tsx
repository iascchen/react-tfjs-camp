import React, { useEffect, useState } from 'react'
import { Tensor } from '@tensorflow/tfjs'
import { Card } from 'antd'
import { Axis, Chart, Geom, Legend, Tooltip } from 'bizcharts'

import { logger } from '../../utils'

const MAX_SAMPLES_COUNT = 100

const prepareData = (_tensor: Tensor, _sampleCount: number = MAX_SAMPLES_COUNT): number[] => {
    if (!_tensor) {
        return []
    }
    const _array = _tensor.dataSync()
    return Array.from(_array ?? []).splice(0, _sampleCount)
}

interface IChartData {
    x: number
    y: number
    type: string
}

interface IProps {
    xDataset: Tensor
    yDataset: Tensor
    pDataset?: Tensor
    sampleCount?: number

    debug?: boolean
}

const CurveVis = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [xData, setXData] = useState<number[]>([])
    const [yData, setYData] = useState<number[]>([])
    const [pData, setPData] = useState<number[]>([])
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
    }, [props.xDataset, sampleCount])

    useEffect(() => {
        if (!props.yDataset) {
            return
        }
        logger('init yDataset ...')

        const _d = prepareData(props.yDataset, sampleCount)
        setYData(_d)
    }, [props.yDataset, sampleCount])

    useEffect(() => {
        if (!props.pDataset) {
            return
        }
        logger('init pDataset ...')

        const _d = prepareData(props.pDataset, sampleCount)
        setPData(_d)
    }, [props.pDataset, sampleCount])

    useEffect(() => {
        logger('init sample data [x,y] ...')
        const _data = xData?.map((v: number, i: number) => {
            return { x: v, type: 'y', y: yData[i] }
        })
        setData(_data)
    }, [xData, yData])

    useEffect(() => {
        logger('init sample data [p] ...')

        const _data: IChartData[] = []
        pData?.forEach((v: number, i: number) => {
            _data.push({ x: xData[i], y: yData[i], type: 'y' })
            _data.push({ x: xData[i], y: v, type: 'p' })
        })
        setData(_data)
    }, [pData])

    /***********************
     * Functions
     ***********************/

    return (
        <Card>
            <Chart height={400} data={data} padding='auto' forceFit>
                <Axis name='X'/>
                <Axis name='Y'/>
                <Legend/>
                <Tooltip/>
                <Geom type='line' position='x*y' size={2} color={'type'} shape={'smooth'}/>
            </Chart>
            Sample count : {props.sampleCount}
            {props.debug ? JSON.stringify(data) : ''}
        </Card>
    )
}

export default CurveVis
