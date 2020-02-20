import React, { useEffect, useState } from 'react'
import { Col, Row } from 'antd'
import {Axis, Chart, Geom, Legend, Tooltip} from 'bizcharts'

import {arrayDispose, ITrainInfo, logger} from '../../../utils'

interface IProps {
    totalEpochs: number
    infos: ITrainInfo[]
    matrix?: string[]
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

        // const _data = props.infos.map(r => {
        //     return { step: r.step, ...r.logs }
        // })

        const _accData: any[] = []
        const _lossData: any[] = []
        props.infos.forEach((v) => {
            _lossData.push({ step: v.step, type: 'loss', value: v.logs.loss })
            _lossData.push({ step: v.step, type: 'val_loss', value: v.logs.val_loss })
            _accData.push({ step: v.step, type: 'acc', value: v.logs.acc })
            _accData.push({ step: v.step, type: 'val_acc', value: v.logs.val_acc })
        })
        setLossData(_lossData)
        setAccData(_accData)

        return () => {
            logger('Dispose data ...')
            arrayDispose(_lossData)
            arrayDispose(_accData)
        }
    }, [props.infos, props.totalEpochs])

    useEffect(() => {
        const _scale = {
            step: { max: props.totalEpochs },
            sync: true
        }
        setScale(_scale)
    }, [props.totalEpochs])

    return (
        <Row>
            <Col span={24}>
                <Chart height={400} data={lossData} padding='auto' scale={scale} forceFit>
                    <Axis name='STEP' />
                    <Axis name='LOSS' />
                    <Legend />
                    <Tooltip/>
                    <Geom type='line' position='step*value' size={2} color={'type'} />
                </Chart>
            </Col>
            <Col span={24}>
                <Chart height={400} data={accData} padding='auto' scale={scale} forceFit>
                    <Axis name='STEP' />
                    <Axis name='ACC' />
                    <Legend />
                    <Tooltip/>
                    <Geom type='line' position='step*value' size={2} color={'type'} />
                </Chart>
            </Col>
            {props.debug ? JSON.stringify(accData) : ''}
            {props.debug ? JSON.stringify(lossData) : ''}
        </Row>
    )
}

export default HistoryWidget
