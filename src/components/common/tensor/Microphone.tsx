import React, { forwardRef, Ref, useEffect, useImperativeHandle, useReducer, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { MicrophoneIterator } from '@tensorflow/tfjs-data/dist/iterators/microphone_iterator'
import { Button, message } from 'antd'

import { ILabelMap, logger } from '../../../utils'
import { ImageNetClasses } from '../../mobilenet/ImageNetClasses'
import TensorImageThumbWidget from './TensorImageThumbWidget'

const VIDEO_SHAPE = [480, 360] // [width, height]
const microphoneConfig: tf.data.MicrophoneConfig = {
    fftSize: 1024,
    columnTruncateLength: 232,
    numFramesPerSpectrogram: 43,
    sampleRateHz: 44100,
    smoothingTimeConstant: 0,
    includeSpectrogram: true,
    includeWaveform: true
}

export interface IMicrophoneHandler {
    capture: () => Promise<tf.TensorContainerObject | void>
}

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor
    isPreview?: boolean
    labelsMap?: ILabelMap

    onSubmit?: (tensor: tf.Tensor) => void
}

const Microphone = (props: IProps, ref: Ref<IMicrophoneHandler>): JSX.Element => {
    const [sLabel, setLabel] = useState<string>()
    const [sPreview, setPreview] = useState<tf.Tensor3D>()

    const [sMic, setMic] = useState<MicrophoneIterator>()

    const [sSwitch, toogleSwitch] = useReducer((x: boolean) => !x, false)

    // const videoRef = useRef<HTMLVideoElement>(null)

    useImperativeHandle(ref, (): IMicrophoneHandler => ({
        capture
    }))

    useEffect(() => {
        let _mic: MicrophoneIterator
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        tf.data.microphone(microphoneConfig).then(
            (mic: MicrophoneIterator) => {
                _mic = mic
                setMic(_mic)
            }
        )

        return () => {
            _mic?.stop()
        }
    }, [])

    useEffect(() => {
        if (!props.prediction) {
            return
        }

        // logger(props.prediction)
        const imagenetRet = props.prediction
        const labelIndex = imagenetRet.arraySync() as number
        logger('labelIndex', labelIndex)
        const label = props.labelsMap ? props.labelsMap[labelIndex] : ImageNetClasses[labelIndex]
        setLabel(`${labelIndex.toString()} : ${label}`)
        imagenetRet.dispose()
    }, [props.prediction])

    const capture = async (): Promise<tf.TensorContainerObject | void> => {
        if (!sMic) {
            return
        }
        const audioData = await sMic.capture() // tensor of shape [43, 232, 1].
        const spectrogramTensor = audioData.spectrogram
        const waveformTensor = audioData.waveform
        return { spectrogram: spectrogramTensor, waveform: waveformTensor }

        // props.isPreview && setPreview(processedImg)
        // return processedImg
    }

    // const handleCapture = async (): Promise<void> => {
    //     await capture()
    // }

    // const handleSubmit = async (): Promise<void> => {
    //     const imgTensor = await capture()
    //     if (imgTensor) {
    //         props.onSubmit && props.onSubmit(imgTensor)
    //     }
    // }

    const handleToggle = (): void => {
        if (!sMic) {
            return
        }

        toogleSwitch()
        if (sSwitch) {
            capture().then(
                (result) => {
                    if (result) {
                        logger(result?.spectrogram)
                        logger(result?.waveform)
                        // props.onSubmit && props.onSubmit(imgTensor)
                    }
                    sMic.next().then()
                },
                (e) => {
                    logger(e)
                    message.error(e.msg)
                }
            )
        } else {
            sMic.stop()
        }
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <div style={{ margin: 16 }}>
                {props.isPreview && (
                    <Button onClick={handleToggle} icon='mic' style={{ width: '30%', margin: '0 10%' }}>
                        Record {sSwitch ? 'ON' : 'Off'}
                    </Button>
                )}
                {/* <Button onClick={handleSubmit} type='primary' style={{ width: '30%', margin: '0 10%' }}>Predict</Button> */}

            </div>
            {props.isPreview && (
                <>
                    <div>Captured Images</div>
                    <div>
                        {sPreview && <TensorImageThumbWidget width={VIDEO_SHAPE[0] / 2} height={VIDEO_SHAPE[1] / 2}
                            data={sPreview}/>}
                    </div>
                </>
            )}
            Prediction Result : {sLabel}
        </>
    )
}

export default forwardRef(Microphone)
