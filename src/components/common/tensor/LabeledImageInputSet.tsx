import React, { useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Icon } from 'antd'
import { FormComponentProps } from 'antd/es/form'

import { ILabeledImageFileJson, logger } from '../../../utils'
import LabeledImageInput from './LabeledImageInput'

const formItemLayout = {
    labelCol: {
        xs: { span: 24 },
        sm: { span: 4 }
    },
    wrapperCol: {
        xs: { span: 24 },
        sm: { span: 20 }
    }
}

const formItemLayoutWithOutLabel = {
    wrapperCol: {
        xs: { span: 24, offset: 0 },
        sm: { span: 20, offset: 4 }
    }
}

export const formatDataJson = (values: any): ILabeledImageFileJson => {
    logger(values)
    const { keys: _keys, labeledImageList } = values
    const ret = _keys.map((key: number) => JSON.parse(labeledImageList[key]))

    const formModel = { labeledImageSetList: ret }
    // logger(JSON.stringify(formModel))
    return formModel
}

interface IProps extends FormComponentProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor

    onSave?: (value: any) => void
}

const LabeledImageInputSet = (props: IProps): JSX.Element => {
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
            console.log(err, values)
            props.onSave && props.onSave(formatDataJson(values))
        })
    }

    /***********************
     * Render
     ***********************/

    const { form } = props
    const { getFieldDecorator, getFieldValue } = form

    getFieldDecorator('keys', { initialValue: keys || [] })
    const _keys = getFieldValue('keys')
    // logger('_keys', _keys, dyncId)

    const formItems = _keys.map((k: number, index: number) => (
        <Form.Item key={k}
            {...(index === 0 ? formItemLayout : formItemLayoutWithOutLabel)}
            label={index === 0 ? 'Label : ' : ''}>
            <Col span={20}>
                <Form.Item>
                    {getFieldDecorator(`labeledImageList[${k.toString()}]`, { initialValue: '{}' })(
                        <LabeledImageInput />
                    )}
                </Form.Item>
            </Col>
            <Col span={2}/>
            <Col span={2}>
                {_keys.length > 1 ? (
                    <Icon className='dynamic-delete-button' type='minus-circle-o'
                        onClick={() => remove(k)}
                    />
                ) : null}
            </Col>
        </Form.Item>
    ))

    return (
        <Card>
            {/* [{_keys.join(',')}], {dyncId} */}
            <Form onSubmit={handleSubmit}>
                {formItems}
                <Form.Item {...formItemLayoutWithOutLabel} style={{ margin: 16 }}>
                    <Button type='dashed' onClick={add} style={{ width: '35%', marginRight: '10%' }}>
                        <Icon type='plus-circle'/> Add
                    </Button>

                    <Button type='primary' htmlType='submit' style={{ width: '35%', marginRight: '10%' }}>
                        Submit
                    </Button>
                </Form.Item>
            </Form>
            <a ref={downloadRef}/>
        </Card>
    )
}

export default Form.create<IProps>({ name: 'labeled-images-form' })(LabeledImageInputSet)
