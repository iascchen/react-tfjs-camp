import React, { FormEvent, useRef, useState } from 'react'
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

const LabeledImageWidget = (props: IProps): JSX.Element => {
    const [dyncId, setDyncId] = useState<number>(0)
    const [keys, setKeys] = useState<number[]>([])

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

    const handleJsonSave = (e: FormEvent): void => {
        e.preventDefault()
        props.form.validateFields((err, values) => {
            if (!err) {
                // logger(values)
                const obj = formatDataJson(values)
                const a = downloadRef.current
                if (a) {
                    const blob = new Blob(
                        [JSON.stringify(obj, null, 2)],
                        { type: 'application/json' })
                    const blobUrl = window.URL.createObjectURL(blob)
                    logger(blobUrl)

                    // console.log(a)
                    const filename = 'labeledImages.json'
                    a.href = blobUrl
                    a.download = filename
                    a.click()
                    window.URL.revokeObjectURL(blobUrl)
                }
            }
        })
    }

    // const handleJsonChange = async ({ file }: UploadChangeParam): Promise<void> => {
    //     logger('handleFileChange', file.name)
    //
    //     const buffer = await getUploadFileArray(file.originFileObj)
    //     const data = JSON.parse(buffer.toString())
    //
    //     const _labeledImageList = data.labeledImageList
    //     if (!_labeledImageList) {
    //         return
    //     }
    //
    //     const _keys = _labeledImageList.map((item: any, index: number) => index)
    //     const imageListStr = _labeledImageList.map((item: any) => {
    //         return JSON.stringify(item)
    //     })
    //     // setLabeledImageList(_labeledImageList)
    //     setDyncId(_keys.length)
    //     setKeys(_keys)
    //
    //     props.form.setFieldsValue(
    //         {
    //             keys: _keys,
    //             labeledImageList: imageListStr
    //         }
    //     )
    // }
    //
    // const handleUpload = async (file: RcFile): Promise<string> => {
    //     // logger(file)
    //     return getUploadFileBase64(file)
    // }

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
            label={index === 0 ? 'Labeled Images : ' : ''}>
            <Col span={20}>
                <Form.Item>
                    {getFieldDecorator(`labeledImageList[${k}]`, { initialValue: '{}' })(
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

                    {/* TODO: Disable Load button */}
                    {/* <Upload onChange={handleJsonChange} action={handleUpload} showUploadList={false}> */}
                    {/*    <Button style={{ width: '20%', marginRight: '10%' }}> */}
                    {/*        <Icon type='upload'/> Load */}
                    {/*    </Button> */}
                    {/* </Upload> */}

                    <Button style={{ width: '35%', marginRight: '10%' }} onClick={handleJsonSave}>
                        <Icon type='save'/> Save
                    </Button>

                    <Button type='primary' htmlType='submit' style={{ width: '80%', marginRight: '10%' }}>
                        Submit
                    </Button>
                </Form.Item>
            </Form>
            <a ref={downloadRef}/>
        </Card>
    )
}

export default Form.create<IProps>({ name: 'labeled-images-form' })(LabeledImageWidget)
