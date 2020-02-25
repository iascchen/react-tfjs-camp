import React, { ChangeEvent, useEffect, useState } from 'react'
import { Card, Icon, Input, message, Modal, Upload } from 'antd'

import { getBase64, logger } from '../../../utils'
import { RcFile, UploadChangeParam, UploadFile } from 'antd/es/upload/interface'

const MAX_FILES = 50

export interface ILabeledImagesItem {
    label?: string
    imageList?: UploadFile[] // with base64 image data
}

interface IProps {
    index?: number
    value?: string
    onChange?: (value: string) => void
}

const LabeledImageItem = (props: IProps): JSX.Element => {
    const [previewImage, setPreviewImage] = useState<string>()
    const [modelDisplay, setModalDispaly] = useState(false)

    const [initValue, setInitValue] = useState<ILabeledImagesItem>()
    const [label, setLabel] = useState<string>('')
    const [imageList, setImageList] = useState<UploadFile[]>([])

    useEffect(() => {
        if (props.value) {
            const obj: ILabeledImagesItem = JSON.parse(props.value)
            setInitValue(obj)
        }
    }, [props.value])

    useEffect(() => {
        const { onChange } = props
        if (onChange) {
            const ret = { index: props.index, label, imageList }

            onChange(JSON.stringify(ret))
        }
    }, [label, imageList])

    const handleCancel = (): void => {
        setModalDispaly(false)
    }

    const handlePreview = async (file: UploadFile): Promise<void> => {
        // logger('handlePreview', file)
        let imgSrc = file.url ?? file.preview
        if (!imgSrc) {
            const result = await getBase64(file.originFileObj)
            file.preview = result
            imgSrc = file.preview
        }

        if (imgSrc) {
            setPreviewImage(imgSrc)
            setModalDispaly(true)
        }
    }

    const handleLabelChange = (e: ChangeEvent<HTMLInputElement>): void => {
        const value: string = e.target.value
        setLabel(value)
    }

    const handleImageChange = async ({ fileList }: UploadChangeParam): Promise<void> => {
        logger('handleImageChange', fileList.length)

        if (fileList.length > MAX_FILES) {
            fileList.splice(MAX_FILES)
            await message.error(`All images are stored in memory, so each label ONLY contains < ${MAX_FILES.toString()} files`)
        }
        setImageList(fileList)
    }

    const handleUpload = async (file: RcFile): Promise<string> => {
        // logger(file)
        return getBase64(file)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Card>
            <Input onChange={handleLabelChange} defaultValue={initValue?.label} placeholder={'Label. such as: cat, dog...'}/>
            <Upload.Dragger action={handleUpload} onChange={handleImageChange} onPreview={handlePreview}
                defaultFileList={initValue?.imageList} fileList={imageList} multiple
                className='upload-list-inline' listType='picture-card'>
                <p className='ant-upload-drag-icon'>
                    <Icon type='inbox' />
                </p>
                <p className='ant-upload-text'>Click or drag files to this area to upload</p>
                <p className='ant-upload-hint'>Support for a single or bulk upload.</p>
                <p className='ant-upload-hint'>Should be less than {MAX_FILES} files.</p>
            </Upload.Dragger>

            <Modal visible={modelDisplay} footer={null} onCancel={handleCancel}>
                <img alt='example' style={{ width: '100%' }} src={previewImage} />
            </Modal>
        </Card>
    )
}

export default LabeledImageItem
