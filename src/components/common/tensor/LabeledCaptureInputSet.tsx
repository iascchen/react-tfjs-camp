import React, { useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Col, Form, Icon } from 'antd'
import { FormComponentProps } from 'antd/es/form'

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
    const { keys: _keys, labeledImageList } = values
    const ret = _keys.map((key: number) => labeledImageList[key])

    const formModel = { labeledImageSetList: ret }
    return formModel
}

interface IProps extends FormComponentProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor

    onSave?: (value: any) => void
    onCapture?: (label: string) => Promise<ILabeledImage | void>
}

const LabeledCaptureInputSet = (props: IProps): JSX.Element => {
    const [dyncId, setDyncId] = useState<number>(0)
    const [keys] = useState<number[]>([])

    const downloadRef = useRef<HTMLAnchorElement>(null)

    const remove = (k: number): void => {
        // can use data-binding to get
        const { form } = props
        const keys = form?.getFieldValue('keys')
        // We need at least one item
        if (keys.length === 1) {
            return
        }

        // can use data-binding to set
        form?.setFieldsValue({
            keys: keys.filter((key: number) => key !== k)
        })
    }

    const add = (): void => {
        // can use data-binding to get
        const { form } = props
        const _keys = form?.getFieldValue('keys')
        const nextKeys = _keys.concat(dyncId)
        setDyncId(dyncId + 1)

        // can use data-binding to set
        // important! notify form to detect changes
        form?.setFieldsValue({
            keys: nextKeys
        })
    }

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
        e.preventDefault()
        props.form.validateFields((err, values) => {
            logger(err, values)
            props.onSave && props.onSave(formatDataJson(values))
        })
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

    const { form } = props
    const { getFieldDecorator, getFieldValue } = form

    getFieldDecorator('keys', { initialValue: keys || [] })
    const _keys = getFieldValue('keys')

    const formItems = _keys.map((k: number, index: number) => (
        <Form.Item key={k} {...formItemLayout } >
            <Col span={22}>
                <Form.Item>
                    {getFieldDecorator(`labeledImageList[${k.toString()}]`)(
                        <LabeledCaptureInput onCapture={handleCapture} />
                    )}
                </Form.Item>
            </Col>
            <Col span={1}/>
            <Col span={1}>
                {_keys.length > 1 ? (
                    <Icon className='dynamic-delete-button' type='minus-circle-o'
                        onClick={() => remove(k)}
                    />
                ) : null}
            </Col>
        </Form.Item>
    ))

    return (
        <>
            <Form onSubmit={handleSubmit}>
                {formItems}
                <Form.Item {...formItemLayout}>
                    <Button type='primary' htmlType='submit' style={{ width: '30%', margin: '0 10%' }}>
                        Push to Train Set
                    </Button>
                    <Button type='dashed' onClick={add} style={{ width: '30%', margin: '0 10%' }}>
                        <Icon type='plus-circle'/> Add
                    </Button>
                </Form.Item>
            </Form>
            <a ref={downloadRef}/>
        </>
    )
}

export default Form.create<IProps>({ name: 'labeled-images-form' })(LabeledCaptureInputSet)
