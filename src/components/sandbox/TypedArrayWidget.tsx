import React from 'react'
import * as tf from '@tensorflow/tfjs'
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

    const t3d = tf.tensor3d([0.1, 0.9, 0.7], [1, 1, 3], 'float32')
    logger(t3d.dataSync())

    const f32Buf = new Float32Array(t3d.dataSync())
    // console.log(f32Buf.length)
    const ui8Buf = new Uint8Array(f32Buf.buffer)
    // console.log(ui8Buf.length)
    const t3dBase64 = Buffer.from(ui8Buf).toString('base64')

    const buf = Buffer.from(t3dBase64, 'base64')
    const ui8Buf2 = new Uint8Array(buf)
    // console.log(ui8Buf.length)
    const f32Buf2 = new Float32Array(ui8Buf2.buffer)
    // console.log(f32Buf.length)
    const t3dNew = tf.tensor3d(f32Buf2, [1, 1, 3], 'float32')
    logger(t3dNew.dataSync())

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
