import React, { useEffect, useRef, useState, useReducer } from 'react'
import { Col, Row } from 'antd'
import { ITrainInfo } from '../../../utils'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

interface IProps {
    logMsg: ITrainInfo | undefined

    debug?: boolean
}

const logs = {
    history: {
        loss: [] as number[],
        val_loss: [] as number[],
        acc: [] as number[],
        val_acc: [] as number[]
    }
}

const HistoryWidgetTfvis = (props: IProps): JSX.Element => {
    const [logData, setLogData] = useState(logs)

    const [ignored, forceUpdate] = useReducer(x => x + 1, 0)

    const lossCanvasRef = useRef<HTMLDivElement>(null)
    const accuracyCanvasRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!props.logMsg || !props.logMsg.logs) {
            return
        }
        const _log = props.logMsg.logs
        logData.history.loss.push(_log.loss)
        logData.history.val_loss.push(_log.val_loss)
        logData.history.acc.push(_log.acc)
        logData.history.val_acc.push(_log.val_acc)

        setLogData(logData)
        forceUpdate()
    }, [props.logMsg])

    useEffect(() => {
        if (!logData) {
            return
        }

        tfvis.show.history(lossCanvasRef.current, logData, ['loss', 'val_loss'])
        tfvis.show.history(accuracyCanvasRef.current, logData, ['acc', 'val_acc'])
    }, [ignored])

    return (
        <Row>
            <Col span={24}>
                <div ref={lossCanvasRef} />
            </Col>
            <Col span={24}>
                <div ref={accuracyCanvasRef} />
            </Col>
        </Row>
    )
}

export default HistoryWidgetTfvis
