import React, { FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Form, Icon, Col } from 'antd'
import { FormComponentProps } from 'antd/es/form'

import { logger } from '../../../utils'
import LabeledImageItem, { ILabeledImagesItem } from './LabeledImageItem'

interface IProps extends FormComponentProps{
    model?: tf.LayersModel
    prediction?: tf.Tensor
    // onSubmit?: (tensor: tf.Tensor) => void

    onSubmit?: (value: object) => void
    labeledImageList: ILabeledImagesItem[]
}

const IMAGE_SIZE = 224

const LabeledImageWidget = (props: IProps): JSX.Element => {
    const [dyncId, setDyncId] = useState<number>(0)
    const [keys, setKeys] = useState<number[]>([])
    const [labeledImageList, setLabeledImageList] = useState<ILabeledImagesItem[]>([])

    // useEffect(() => {
    //     if (!props.prediction) {
    //         return
    //     }
    //
    //     const labelIndex = props.prediction.arraySync() as number
    //     logger('labelIndex', labelIndex)
    //     const _label = ImagenetClasses[labelIndex]
    // }, [props.prediction])

    useEffect(() => {
        props.labeledImageList && setDyncId(props.labeledImageList?.length)
    }, [props.labeledImageList])

    useEffect(() => {
        const _keys = labeledImageList ? labeledImageList.map((item, index) => index) : []
        setKeys(_keys)
    }, [labeledImageList])

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
        console.log('_keys', _keys, dyncId)
        const nextKeys = _keys.concat(dyncId)
        setDyncId(dyncId + 1)
        console.log('nextKeys', nextKeys, dyncId)
        // can use data-binding to set
        // important! notify form to detect changes
        form?.setFieldsValue({
            keys: nextKeys
        })
    }

    const handleSubmit = (e: FormEvent): void => {
        e.preventDefault()
        props.form.validateFields((err, values) => {
            if (!err) {
                // console.log(values)

                const { keys, labeledImageList } = values
                const ret = keys.map((key: number) => JSON.parse(labeledImageList[key]))

                const formModel = {
                    labeledImageList: ret
                }

                // console.log(JSON.stringify(formModel))
                props.onSubmit && props.onSubmit(formModel)
            }
        })
    }

    /***********************
     * Render
     ***********************/

    const { form } = props
    const { getFieldDecorator, getFieldValue } = form

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

    getFieldDecorator('keys', { initialValue: keys || [] })
    const _keys = getFieldValue('keys')
    console.log('_keys', _keys, dyncId)

    const formItems = _keys.map((k: number, index: number) => (
        <Form.Item key={k}
            {...(index === 0 ? formItemLayout : formItemLayoutWithOutLabel)}
            label={index === 0 ? 'Labeled Images' : ''}>
            <Col span={20}>
                <Form.Item>
                    {getFieldDecorator(`labeledImageList[${k}]`, {
                        initialValue: labeledImageList?.[k] ? JSON.stringify(labeledImageList[k]) : '{}'
                    })(
                        <LabeledImageItem />
                    )}
                </Form.Item>
            </Col>
            <Col span={2} />
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
            {JSON.stringify(_keys)}
            <Form onSubmit={handleSubmit}>
                {formItems}
                <Form.Item {...formItemLayoutWithOutLabel} style={{ margin: 16 }}>
                    <Button type='dashed' size={'large'} onClick={add} style={{ width: '30%', marginRight: '10%' }}>
                        <Icon type='plus-circle' /> Add
                    </Button>

                    <Button type='primary' size={'large'} htmlType='submit' style={{ width: '30%', marginRight: '10%' }}>
                        Submit
                    </Button>
                </Form.Item>
            </Form>
        </Card>
    )
}

export default Form.create<IProps>({ name: 'labeled-images-form' })(LabeledImageWidget)
