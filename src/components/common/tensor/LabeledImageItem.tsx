import React, { ChangeEvent, useEffect, useReducer, useRef, useState } from 'react'
import { Card, Icon, Input, message, Modal, Upload } from 'antd'
import { RcFile, UploadChangeParam, UploadFile } from 'antd/es/upload/interface'

import { getUploadFileBase64, logger } from '../../../utils'

const { Dragger } = Upload

const MAX_FILES = 3

export interface ILabeledImage {
    uid: string
    name: string
    status: string
    img: string // Base64 Array
}

export interface ILabeledImageSet {
    label?: string
    imageDataList?: ILabeledImage[]
}

interface IProps {
    index?: number
    value?: string

    onChange?: (value: string) => void
}

const checkUnload = (fileList: UploadFile[]): number => {
    let unload: number = fileList.length
    fileList.forEach(item => {
        // console.log(item.status)
        if (item.status === 'done') {
            unload--
        }
    })
    logger('waiting checkUnload : ', fileList.length, unload)
    return unload
}

const LabeledImageItem = (props: IProps): JSX.Element => {
    const [previewImage, setPreviewImage] = useState<string>()
    const [modelDisplay, setModalDispaly] = useState(false)

    const [label, setLabel] = useState<string>('')
    const [imageList, setImageList] = useState<UploadFile[]>([])

    const [waitingPush, forceWaitingPush] = useReducer((x: number) => x + 1, 0)

    const labelRef = useRef<Input>(null)

    useEffect(() => {
        if (!props.value) {
            return
        }

        logger('Can not init from stored file')
        // TODO Cannot init AntD <Upload> by binary data
        // const { label: _label, imageDataList: _imageDataList } = JSON.parse(props.value) as ILabeledImageSet
        // if (_label && labelRef.current) {
        //     labelRef.current.setValue(_label)
        // }
        //
        // if (_imageDataList) {
        //     const _imageList = _imageDataList.map(item => {
        //         const { uid, name, status, data } = item
        //         const file = new File ([data], name)
        //         return { uid, name, status, file } as UploadFile
        //     })
        //     setImageList(_imageList)
        // }
    }, [props.value])

    useEffect(() => {
        const timer = setInterval(async (): Promise<void> => {
            logger('Waiting upload...')
            if (checkUnload(imageList) > 0) {
                forceWaitingPush()
            } else {
                clearInterval(timer)

                const imgDataList = []
                for (let i = 0; i < imageList.length; i++) {
                    const { uid, name, status, originFileObj } = imageList[i]
                    const result = await getUploadFileBase64(originFileObj)
                    const imgData = { uid, name, status, data: result }
                    imgDataList.push(imgData)
                }
                const itemData = { label, imgDataList }

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
        setModalDispaly(false)
    }

    const handlePreview = async (file: UploadFile): Promise<void> => {
        // logger('handlePreview', file)
        let imgSrc = file.url ?? file.preview
        if (!imgSrc) {
            console.log('originFileObj', file.originFileObj)
            const result = await getUploadFileBase64(file.originFileObj)
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
        // logger('handleImageChange', fileList.length)

        if (fileList.length > MAX_FILES) {
            fileList.splice(MAX_FILES)
            await message.error(`All images are stored in memory, so each label ONLY contains < ${MAX_FILES.toString()} files`)
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
            {props.index}
            <Input onChange={handleLabelChange} defaultValue={label} ref={labelRef}
                placeholder={'Label. such as: cat, dog...'} />
            <Dragger action={handleUpload} onChange={handleImageChange} onPreview={handlePreview}
                fileList={imageList} multiple
                className='upload-list-inline' listType='picture-card'>
                <p className='ant-upload-drag-icon'>
                    <Icon type='inbox' />
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

export default LabeledImageItem
