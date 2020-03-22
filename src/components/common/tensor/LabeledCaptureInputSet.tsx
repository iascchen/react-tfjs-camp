import React, { useRef } from 'react'
import { Button, Col, Form, message, Row } from 'antd'
import { MinusCircleOutlined, PlusCircleOutlined } from '@ant-design/icons'

import { formItemLayout } from '../../../constant'
import { ILabeledImage, logger } from '../../../utils'
import LabeledCaptureInput from './LabeledCaptureInput'

interface IProps {
    onSave?: (value: any) => void
    onCapture?: (label: string) => Promise<ILabeledImage | void>
}

const LabeledCaptureInputSet = (props: IProps): JSX.Element => {
    const downloadRef = useRef<HTMLAnchorElement>(null)
    const [form] = Form.useForm()

    const handleSubmit = (values: any): void => {
        logger('handleSubmit', values)
        const labeledList = values.labeledImageList
        if (labeledList.length >= 2) {
            props.onSave && props.onSave({ labeledImageSetList: labeledList })
        } else {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.error('At least have 2 lebels!')
        }
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
                            {fields.map((field) => (
                                <Form.Item required={false} key={field.key} {...formItemLayout} >
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
