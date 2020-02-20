import React from 'react'
import { logger } from '../../utils'

const TypedArrayWidget = (): JSX.Element => {
    const a = [1, 2, 3, 4, 5, 6]
    logger('a', a)

    const c = a.map((r, idx) => {
        logger(r, idx)
        return r + 1
    })
    logger('c', c)

    const b = new Int8Array(a)
    logger('b', b)

    const d = b.map((r, idx) => {
        logger(r, idx)
        return r + 1
    })
    logger('d', Array.from(d))

    return (
        <div>
            <div>{
                c.map((r, idx) => {
                    return <div key={idx}>{r}</div>
                })
            }</div>

            <div>{
                Array.from(d).map((r, idx) => {
                    return <div key={idx}>{r}</div>
                })
            }</div>
        </div>
    )
}

export default TypedArrayWidget
