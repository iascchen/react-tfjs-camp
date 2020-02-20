import React, { useEffect, useState } from 'react'
import { Col, Row } from 'antd'
import { Axis, Chart, Geom, Legend, Tooltip } from 'bizcharts'

import { arrayDispose, ITrainInfo, logger } from '../../../utils'

interface IProps {
    totalIterations: number
    infos: ITrainInfo[]
    debug?: boolean
}

const HistoryWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [accData, setAccData] = useState()
    const [lossData, setLossData] = useState()
    const [scale, setScale] = useState()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data ...')

        // TODO : need more performance
        const _accData: any[] = []
        const _lossData: any[] = []
        props.infos.forEach((v) => {
            _lossData.push({ iteration: v.iteration, type: 'loss', value: v.logs.loss })
            _lossData.push({ iteration: v.iteration, type: 'val_loss', value: v.logs.val_loss })
            _accData.push({ iteration: v.iteration, type: 'acc', value: v.logs.acc })
            _accData.push({ iteration: v.iteration, type: 'val_acc', value: v.logs.val_acc })
        })
        setLossData(_lossData)
        setAccData(_accData)

        return () => {
            logger('Dispose data ...')
            arrayDispose(_lossData)
            arrayDispose(_accData)
        }
    }, [props.infos])

    useEffect(() => {
        const _scale = {
            iteration: { max: props.totalIterations }
        }
        setScale(_scale)
    }, [props.totalIterations])

    return (
        <Row>
            <Col span={24}>
                <Chart height={400} data={lossData} padding='auto' scale={scale} forceFit>
                    <Axis name='ITERATION'/>
                    <Axis name='LOSS'/>
                    <Legend/>
                    <Tooltip/>
                    <Geom type='line' position='iteration*value' size={2} color={'type'}/>
                </Chart>
            </Col>
            <Col span={24}>
                <Chart height={400} data={accData} padding='auto' scale={scale} forceFit>
                    <Axis name='ITERATION'/>
                    <Axis name='ACC'/>
                    <Legend/>
                    <Tooltip/>
                    <Geom type='line' position='iteration*value' size={2} color={'type'}/>
                </Chart>
            </Col>
            {props.debug && JSON.stringify(accData)}
            {props.debug && JSON.stringify(lossData)}
            {props.debug && props.totalIterations}
        </Row>
    )
}

export default HistoryWidget
