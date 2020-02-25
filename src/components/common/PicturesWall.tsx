import React, { useState } from 'react'
import { Icon, Modal, Upload } from 'antd'
import { UploadFile, UploadChangeParam, RcFile } from 'antd/es/upload/interface'
import { logger, getUploadFileBase64 } from '../../utils'

interface IProps {
    onPreview?: (file: string) => void
}

const PicturesWall = (props: IProps): JSX.Element => {
    const [previewImage, setPreviewImage] = useState<string>()
    const [imageList, setImageList] = useState<UploadFile[]>([])
    const [modelDisplay, setModalDispaly] = useState(false)

    const handleCancel = (): void => {
        setModalDispaly(false)
    }

    const handlePreview = async (file: UploadFile): Promise<void> => {
        // logger('handlePreview', file)

        let imgSrc = file.url ?? file.preview
        if (!imgSrc) {
            const result = await getUploadFileBase64(file.originFileObj)
            file.preview = result
            imgSrc = file.preview
        }

        if (imgSrc) {
            setPreviewImage(imgSrc)
            // setModalDispaly(true)
            props.onPreview && props.onPreview(imgSrc)
        }
    }

    const handleChange = ({ fileList }: UploadChangeParam): void => {
        // logger('handleChange', fileList)
        setImageList(fileList)
    }

    const handleUpload = async (file: RcFile): Promise<string> => {
        // logger(file)
        return getUploadFileBase64(file)
    }

    const uploadButton = (
        <div>
            <Icon type='plus' />
            <div className='ant-upload-text'>Upload</div>
        </div>
    )

    return (
        <div className='clearfix'>
            <Upload action={handleUpload} fileList={imageList} onPreview={handlePreview} onChange={handleChange}
                listType='picture-card'>
                {imageList.length >= 8 ? null : uploadButton}
            </Upload>

            <Modal visible={modelDisplay} footer={null} onCancel={handleCancel}>
                <img alt='example' style={{ width: '100%' }} src={previewImage} />
            </Modal>
        </div>
    )
}

export default PicturesWall
