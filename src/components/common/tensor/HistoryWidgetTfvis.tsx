import React, { useEffect, useRef } from 'react'
// import { show } from '@tensorflow/tfjs-vis'
import { Col, Row } from 'antd'

import { ITrainInfo } from '../../../utils'

interface IProps {
    totalEpochs: number
    infos: ITrainInfo[]
    matrix?: string[]
}

const HistoryWidgetTfvis = (props: IProps): JSX.Element => {
    const lossCanvasRef = useRef<HTMLCanvasElement>(null)
    const accuracyCanvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        // const trainLogs = props.infos.map(r => {
        //     return r.logs
        // })
        // const log = trainLogs

        // show.history(lossCanvasRef.current, log, ['loss', 'val_loss']).then(
        //     () => {
        //         // do nothing
        //     }
        // )
        // show.history(accuracyCanvasRef.current, log, ['acc', 'val_acc']).then(
        //     () => {
        //     // do nothing
        // })
    }, [props.infos])

    return (
        <Row>
            <Col span={24}>
                <canvas height={400} ref={lossCanvasRef} />
            </Col>
            <Col span={24}>
                <canvas height={400} ref={accuracyCanvasRef} />
            </Col>
            {JSON.stringify(props.infos)}
        </Row>
    )
}

export default HistoryWidgetTfvis
