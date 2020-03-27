import React, { useEffect, useState } from 'react'
import { STATUS, logger, fetchResource } from '../../utils'

const resultURL = '/data/t10k-labels-idx1-ubyte.gz'

const FetchWidget = (): JSX.Element => {
    const [status, setStatus] = useState<STATUS>()
    const [status2, setStatus2] = useState<STATUS>()
    const [buffer, setBuffer] = useState<ArrayBuffer>()
    const [buffer2, setBuffer2] = useState<ArrayBuffer>()

    useEffect(() => {
        setStatus(STATUS.WAITING)
        fetchResource(resultURL).then(
            (result) => {
                logger(result)
                setBuffer(result)
                setStatus(STATUS.LOADED)
            },
            (error) => {
                // ignore
                logger(error)
            })
    }, [])

    useEffect(() => {
        setStatus2(STATUS.WAITING)
        fetchResource(resultURL, true).then(
            (result) => {
                logger(result)
                setBuffer2(result)
                setStatus2(STATUS.LOADED)
            },
            () => {
                // ignore
            })
    }, [])

    return (
        <div>
            <div>Fetch {status} : {resultURL}</div>
            <div>FileSize : {buffer?.byteLength}</div>
            <hr/>
            <div>Unziped Fetch {status2} : {resultURL}</div>
            <div>FileSize : {buffer2?.byteLength}</div>
        </div>
    )
}

export default FetchWidget
