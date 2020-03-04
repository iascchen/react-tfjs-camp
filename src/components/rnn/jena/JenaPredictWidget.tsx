import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, message, Select } from 'antd'

import { logger, STATUS } from '../../../utils'

import { JenaWeatherData } from './dataJena'
import { trainModel } from './modelJena'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select

const TIME_SPAN = ['Full', 'Year', 'Month', 'Week', 'TenDays', 'Day', 'hour']

interface IProps{
    model?: tf.LayersModel
    data?: JenaWeatherData

    // onChange?: (model: JenaWeatherData) => void
}

const JenaPredictWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTimeSpan, setTimeSpan] = useState<tf.Tensor>()
    const [sSeries1, setSeries1] = useState<tf.Tensor>()
    const [sSeries2, setSeries2] = useState<tf.Tensor>()
    const [sNormalize, setNormalize] = useState<boolean>()
    const [sScatter, setScatter] = useState<tf.Tensor>()

    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()

    const [sEpochs, setEpochs] = useState<number>(10)

    const dataChartRef = useRef<HTMLDivElement>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')
    }, [])

    useEffect(() => {
        props.data && setDataHandler(props.data)
    }, [props.data])

    useEffect(() => {
        props.model && setModel(props.model)
    }, [props.model])

    // const surface = tfvis.visor().surface({ tab: 'Training', name: 'Model Training' })

    // const train = async (): Promise<void> => {
    //     if (!sModel || !sDataHandler) {
    //         // eslint-disable-next-line @typescript-eslint/no-floating-promises
    //         message.warn('Data and model are not ready')
    //         return
    //     }
    //
    //     setStatus(STATUS.TRAINING)
    //     logger('train...')
    //
    //     const lookBack = 10 * 24 * 6 // Look back 10 days.
    //     const step = 6 // 1-hour steps.
    //     const delay = 24 * 6 // Predict the weather 1 day later.
    //     const batchSize = 128
    //     const normalize = true
    //     const includeDateTime = false
    //
    //     console.log('Starting model training...')
    //     const epochs = sEpochs
    //     await trainModel(sModel, sDataHandler, normalize, includeDateTime,
    //         lookBack, step, delay, batchSize, epochs,
    //         tfvis.show.fitCallbacks(elementRef.current, ['loss', 'val_loss'], {
    //             callbacks: ['onBatchEnd', 'onEpochEnd']
    //         }))
    //     setStatus(STATUS.TRAINED)
    // }
    //
    // const handleTrain = (): void => {
    //     train().then()
    // }
    //
    // const populateSelects = (dataObj): void => {
    //     const columnNames = ['None'].concat(dataObj.getDataColumnNames())
    //     for (const selectSeries of [selectSeries1, selectSeries2]) {
    //         while (selectSeries.firstChild) {
    //             selectSeries.removeChild(selectSeries.firstChild)
    //         }
    //         console.log(columnNames)
    //         for (const name of columnNames) {
    //             const option = document.createElement('option')
    //             option.setAttribute('value', name)
    //             option.textContent = name
    //             selectSeries.appendChild(option)
    //         }
    //     }
    //
    //     if (columnNames.includes('T (degC)')) {
    //         selectSeries1.value = 'T (degC)'
    //     }
    //     if (columnNames.includes('p (mbar)')) {
    //         selectSeries2.value = 'p (mbar)'
    //     }
    //     timeSpanSelect.value = 'week'
    //     dataNormalizedCheckbox.checked = true
    // }

    // const plotData = (): void {
    //     logger('Rendering data plot...');
    //     const {timeSpan, series1, series2, normalize, scatter} = getDataVizOptions();
    //
    //     if (scatter && series1 !== 'None' && series2 !== 'None') {
    //         // Plot the two series against each other.
    //         makeTimeSeriesScatterPlot(series1, series2, timeSpan, normalize);
    //     } else {
    //         // Plot one or two series agains time.
    //         makeTimeSeriesChart(
    //             series1, series2, timeSpan, normalize, dataChartContainer);
    //     }
    //
    //     updateDateTimeRangeSpan(jenaWeatherData);
    //     updateScatterCheckbox();
    //     logStatus('Done rendering chart.');
    // }

    /***********************
     * Render
     ***********************/

    // timeSpan: timeSpanSelect.value,
    //     series1: selectSeries1.value,
    //     series2: selectSeries2.value,
    //     normalize: dataNormalizedCheckbox.checked,
    //     scatter: dataScatterCheckbox.checked

    return (
        <div>
            <Card title='Jena Weather Predict' style={{ margin: '8px' }} size='small'>
                {/* {sModel} */}
                {/* {sDataHandler} */}
                {/* <Select onChange={handleTimeSpanChange} style={{ marginLeft: 16 }}> */}
                {/*    {TIME_SPAN.map((v) => { */}
                {/*        return <Option key={v} value={v}>{v}</Option> */}
                {/*    })} */}
                {/* </Select> */}

                {/* <Select > */}
                {/*    {data-series-1} */}
                {/* </Select> */}

                {/* <Select > */}
                {/*    {data-series-2} */}
                {/* </Select> */}

                {/* <Checkbox> */}
                {/*    {data-normalized} */}
                {/* </Checkbox> */}
            </Card>
            <div ref={dataChartRef} style={{ height: 400 }}/>
        </div>
    )
}

export default JenaPredictWidget
