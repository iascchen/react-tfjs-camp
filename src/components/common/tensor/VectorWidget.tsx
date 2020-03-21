import React, { useEffect, useState } from 'react'
import { Tensor, Tensor2D } from '@tensorflow/tfjs-core'
import { Col, Row } from 'antd'
import { Axis, Chart, Geom, Legend, Tooltip } from 'bizcharts'

import { arrayDispose, logger } from '../../../utils'

interface IProps {
    data: {
        [label: string]: Tensor2D
    }
    predFeature?: Tensor
    debug?: boolean
}

const VectorWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sData, setData] = useState()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data ...')

        const _data: any[] = []
        Object.keys(props.data).forEach(label => {
            const vector = props.data[label]
            const _sum = Array.from(vector.sum(0).dataSync())
            const _max = Array.from(vector.max(0).dataSync())
            const _min = Array.from(vector.min(0).dataSync())
            _sum.forEach((value, index) => {
                _data.push({ index, label, value: _sum[index] })
                _data.push({ index, label: `${label}_max`, value: _max[index] })
                _data.push({ index, label: `${label}_min`, value: _min[index] })
            })
        })

        if (props.predFeature) {
            Array.from(props.predFeature.dataSync()).forEach((value, index) => {
                _data.push({ index, label: 'pred', value })
            })
        }
        setData(_data)

        return () => {
            logger('Dispose data ...')
            arrayDispose(_data)
        }
    }, [props.data, props.predFeature])

    return (
        <Row>
            <Col span={24}>
                <Chart height={400} data={sData} padding='auto' forceFit>
                    <Axis name='index'/>
                    <Axis name='value'/>
                    <Legend/>
                    <Tooltip/>
                    <Geom type='line' position='index*value' size={2} opacity={0.65} color={'label'} shape='smooth'/>
                </Chart>
            </Col>
            {props.debug && JSON.stringify(sData)}
        </Row>
    )
}

export default VectorWidget
