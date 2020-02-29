import React, { useRef } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Col, Form, Row } from 'antd'
import { MinusCircleOutlined, PlusCircleOutlined } from '@ant-design/icons'

import { ILabeledImage, ILabeledImageFileJson, logger } from '../../../utils'
import LabeledCaptureInput from './LabeledCaptureInput'

const formItemLayout = {
    wrapperCol: {
        xs: { span: 24 },
        sm: { span: 24 }
    }
}

export const formatDataJson = (values: any): ILabeledImageFileJson => {
    logger(values)
    const formModel = { labeledImageSetList: values.labeledImageList }
    return formModel
}

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor

    onSave?: (value: any) => void
    onCapture?: (label: string) => Promise<ILabeledImage | void>
}

const LabeledCaptureInputSet = (props: IProps): JSX.Element => {
    // const [dyncId, setDyncId] = useState<number>(0)
    // const [keys] = useState<number[]>([])

    const downloadRef = useRef<HTMLAnchorElement>(null)

    const [form] = Form.useForm()

    const handleSubmit = (values: any): void => {
        logger('handleSubmit', values)
        props.onSave && props.onSave(formatDataJson(values))
    }

    const handleCapture = async (label: string): Promise<ILabeledImage | void> => {
        // push data to LabeledImageWidget
        const { onCapture } = props
        if (onCapture) {
            logger('handleCapture onCapture')
            return onCapture(label)
        }
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <Form form={form} onFinish={handleSubmit}>
                <Form.List name='labeledImageList'>
                    {(fields, { add, remove }) => {
                        return (<>
                            {fields.map((field, index) => (
                                <Form.Item required={false} key={index} {...formItemLayout} >
                                    <Row>
                                        <Col span={22}>
                                            <Form.Item {...field}>
                                                <LabeledCaptureInput onCapture={handleCapture} />
                                            </Form.Item>
                                        </Col>
                                        <Col span={1}/>
                                        <Col span={1}>
                                            {fields.length > 1 ? (
                                                <MinusCircleOutlined className='dynamic-delete-button'
                                                    onClick={() => remove(field.name)}/>
                                            ) : null}
                                        </Col>
                                    </Row>
                                </Form.Item>))}
                            <Form.Item>
                                <Button type='dashed' onClick={() => add()} style={{ width: '30%', margin: '0 10%' }}>
                                    <PlusCircleOutlined/> Add
                                </Button>
                                <Button type='primary' htmlType='submit' style={{ width: '30%', margin: '0 10%' }}>
                                    Push to Train Set
                                </Button>
                            </Form.Item>
                        </>)
                    }}
                </Form.List>

            </Form>
            <a ref={downloadRef}/>
        </>
    )
}

export default LabeledCaptureInputSet
