import React, { ChangeEvent, useEffect, useReducer, useRef, useState } from 'react'
import { Card, Input, message, Modal, Upload } from 'antd'
import { InboxOutlined } from '@ant-design/icons'
import { RcFile, UploadChangeParam, UploadFile } from 'antd/es/upload/interface'

import { checkUploadDone, getUploadFileBase64, ILabeledImage, ILabeledImageSet, logger } from '../../../utils'

const { Dragger } = Upload

const MAX_FILES = 50

interface IProps {
    value?: ILabeledImageSet

    onChange?: (value: string) => void
}

const LabeledImageInput = (props: IProps): JSX.Element => {
    const [previewImage, setPreviewImage] = useState<string>()
    const [modelDisplay, setModalDisplay] = useState(false)

    const [label, setLabel] = useState<string>('')
    const [imageList, setImageList] = useState<UploadFile[]>([])

    const [waitingPush, forceWaitingPush] = useReducer((x: number) => x + 1, 0)

    const labelRef = useRef<Input>(null)

    useEffect(() => {
        if (!props.value) {
            return
        }

        const { label, imageList } = props.value
        label && setLabel(label)

        // imageList && setImageList(imageList)
    }, [props.value])

    useEffect(() => {
        // eslint-disable-next-line @typescript-eslint/no-misused-promises
        const timer = setInterval(async (): Promise<void> => {
            logger('Waiting upload...')
            if (checkUploadDone(imageList) > 0) {
                forceWaitingPush()
            } else {
                clearInterval(timer)

                const imgDataList = []
                for (let i = 0; i < imageList.length; i++) {
                    const _file = imageList[i]
                    const { uid, name, originFileObj } = _file
                    if (!_file.preview) {
                        logger('originFileObj', originFileObj)
                        const result = await getUploadFileBase64(_file.originFileObj)
                        _file.preview = result
                    }

                    const imgData: ILabeledImage = { uid, name, img: _file.preview }
                    imgDataList.push(imgData)
                }
                const itemData: ILabeledImageSet = { label, imageList: imgDataList }

                // push data to LabeledImageWidget
                const { onChange } = props
                if (onChange) {
                    logger('Uploaded')
                    onChange(JSON.stringify(itemData))
                }
            }
        }, 10)

        return () => {
            clearInterval(timer)
        }
    }, [waitingPush])

    const handleCancel = (): void => {
        setModalDisplay(false)
    }

    const handlePreview = async (file: UploadFile): Promise<void> => {
        // logger('handlePreview', file)
        let imgSrc = file.url ?? file.preview
        if (!imgSrc) {
            logger('originFileObj', file.originFileObj)
            const result = await getUploadFileBase64(file.originFileObj)
            file.preview = result
            imgSrc = file.preview
        }

        if (imgSrc) {
            setPreviewImage(imgSrc)
            setModalDisplay(true)
        }
    }

    const handleLabelChange = (e: ChangeEvent<HTMLInputElement>): void => {
        const value: string = e.target.value
        setLabel(value)

        // Must wait until all file status is 'done', then push then to LabeledImageWidget
        forceWaitingPush()
    }

    const handleImageChange = ({ fileList }: UploadChangeParam): void => {
        // logger('handleImageChange', fileList.length)

        if (fileList.length > MAX_FILES) {
            fileList.splice(MAX_FILES)

            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.error(`All images are stored in memory, so each label ONLY contains < ${MAX_FILES.toString()} files`)
        }
        setImageList(fileList)

        // Must wait until all file status is 'done', then push then to LabeledImageWidget
        forceWaitingPush()
    }

    const handleUpload = async (file: RcFile): Promise<string> => {
        // logger(file)
        return getUploadFileBase64(file)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Card>
            <Input onChange={handleLabelChange} defaultValue={label} ref={labelRef}
                placeholder={'Label. such as: cat, dog...'} />
            <Dragger action={handleUpload} onChange={handleImageChange} onPreview={handlePreview}
                fileList={imageList} multiple
                className='upload-list-inline' listType='picture-card'>
                <p className='ant-upload-drag-icon'>
                    <InboxOutlined />
                </p>
                <p className='ant-upload-text'>Click or drag files to this area to upload</p>
                <p className='ant-upload-hint'>Support for a single or bulk upload.</p>
                <p className='ant-upload-hint'>Should be less than {MAX_FILES} files.</p>
            </Dragger>

            <Modal visible={modelDisplay} footer={null} onCancel={handleCancel}>
                <img alt='example' style={{ width: '100%' }} src={previewImage} />
            </Modal>
        </Card>
    )
}

export default LabeledImageInput
