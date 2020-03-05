import React, { useEffect, useRef, useState, useReducer } from 'react'
import { Col, Row } from 'antd'
import { ITrainInfo } from '../../../utils'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

interface IProps {
    logMsg?: ITrainInfo

    debug?: boolean
}

const logs = {
    history: {
        loss: [3] as number[],
        val_loss: [3] as number[],
        acc: [0] as number[],
        val_acc: [0] as number[]
    }
}

const TfvisHistoryWidget = (props: IProps): JSX.Element => {
    const [logData, setLogData] = useState(logs)

    const [ignore, forceUpdate] = useReducer((x: number) => x + 1, 0)

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
    }, [ignore])

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

export default TfvisHistoryWidget
