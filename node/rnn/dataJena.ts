/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Data object for Jena Weather data.
 *
 * The data used in this demo is the
 * [Jena weather archive
 * dataset](https://www.kaggle.com/pankrzysiu/weather-archive-jena).
 *
 * This file is used to load the Jena weather data in both
 * - the browser: see [index.js](./index.js), and
 * - the Node.js backend environment: see [train-rnn.js](./train-rnn.js).
 */

import * as tf from '@tensorflow/tfjs-node'
import { fetchLocal, logger } from '../utils'

const BASE_URL = './public/preload/data'
const LOCAL_JENA_WEATHER_CSV_PATH = `${BASE_URL}/jena_climate_2009_2016.csv`
// const REMOTE_JENA_WEATHER_CSV_PATH =
//     'https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv'

const SAMPLE_OFFSET = 50
const SAMPLE_LEN = 3

interface IParsedDate {
    date: Date
    normalizedDayOfYear: number // normalizedDayOfYear: Day of the year, normalized between 0 and 1.
    normalizedTimeOfDay: number // normalizedTimeOfDay: Time of the day, normalized between 0 and 1.
}

/**
 * A class that fetches and processes the Jena weather archive data.
 *
 * It also provides a method to create a function that iterates over
 * batches of training or validation data.
 */
export class JenaWeatherData {
    dataColumnNames: string[] = []
    dateTimeCol = 0
    tempCol = 0

    numRows = 0
    numColumns = 0
    numColumnsExcludingTarget = 0

    dateTime: Date[] = []
    // Day of the year data, normalized between 0 and 1.
    normalizedDayOfYear: number[] = []
    // Time of the day, normalized between 0 and 1.
    normalizedTimeOfDay: number[] = []

    data: number[][] = [] // Unnormalized data.
    means: number[] = []
    stddevs: number[] = []
    normalizedData: number[][] = []

    dataset?: tf.TensorContainer[]
    sampleData?: tf.TensorContainerObject

    csvLines: string[] = []

    constructor () {
        // this.csvDataset = tf.data.csv(LOCAL_JENA_WEATHER_CSV_PATH)
    }

    loadCsv = async (): Promise<void> => {
        const buffer = await fetchLocal(LOCAL_JENA_WEATHER_CSV_PATH)
        if (!buffer) {
            return
        }

        const csvData = buffer.toString()
        this.csvLines = csvData.split('\n')
    }

    loadDataColumnNames = (): void => {
        // Parse header.
        const columnNames = this.csvLines[0].split(',')
        for (let i = 0; i < columnNames.length; ++i) {
            // Discard the quotes around the column name.
            columnNames[i] = columnNames[i].slice(1, columnNames[i].length - 1)
        }

        this.dateTimeCol = columnNames.indexOf('Date Time')
        // tf.util.assert(this.dateTimeCol === 0, 'Unexpected date-time column index')

        this.dataColumnNames = columnNames.slice(1)
        this.tempCol = this.dataColumnNames.indexOf('T (degC)')
        // tf.util.assert(this.tempCol >= 1, 'Unexpected T (degC) column index')
    }

    load = async (): Promise<void> => {
        // Parse CSV file. will spend 10+ sec
        // const beginMs = performance.now()

        this.dateTime = []
        this.data = [] // Unnormalized data.
        this.normalizedDayOfYear = [] // Day of the year data, normalized between 0 and 1.
        this.normalizedTimeOfDay = [] // Time of the day, normalized between 0 and 1.

        for (let i = 1; i < this.csvLines.length; ++i) {
            const line = this.csvLines[i].trim()
            if (line.length === 0) {
                continue
            }
            const items = line.split(',')
            const parsed = this.parseDateTime_(items[0])
            const newDateTime: Date = parsed.date
            if (this.dateTime.length > 0 &&
                newDateTime.getTime() <=
                this.dateTime[this.dateTime.length - 1].getTime()) {
            }

            this.dateTime.push(newDateTime)
            this.data.push(items.slice(1).map(x => +x))
            this.normalizedDayOfYear.push(parsed.normalizedDayOfYear)
            this.normalizedTimeOfDay.push(parsed.normalizedTimeOfDay)

            if ((i % 100) === 0) {
                logger('.')
            }
        }
        this.numRows = this.data.length
        this.numColumns = this.data[0].length
        this.numColumnsExcludingTarget = this.data[0].length - 1

        logger(`this.numColumnsExcludingTarget = ${this.numColumnsExcludingTarget}`)

        await this.calculateMeansAndStddevs_()

        // logger('spend time: ', (performance.now() - beginMs) / 1000)
        // logger(this.normalizedData)
    }

    // loadDataColumnNames = async (): Promise<void> => {
    //     // Parse header.
    //     const columnNames = await this.csvDataset.columnNames()
    //     this.dataColumnNames = columnNames.slice(1)
    //     this.tempCol = 0
    //     // logger(columnNames)
    // }

    // load = async (): Promise<void> => {
    //     // // will spend 130+ sec
    //     // // Headers
    //     // const cName = ['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    //     //     'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
    //     //     'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
    //
    //     const beginMs = performance.now()
    //
    //     // const sampleObjs = csvDataset.skip(SAMPLE_OFFSET).take(SAMPLE_LEN)
    //
    //     this.dateTime = []
    //     this.data = [] // Unnormalized data.
    //     this.normalizedDayOfYear = [] // Day of the year data, normalized between 0 and 1.
    //     this.normalizedTimeOfDay = [] // Time of the day, normalized between 0 and 1.
    //
    //     // await allData.forEachAsync((row: any) => {
    //     await this.csvDataset.forEachAsync((row: any) => {
    //         const rowValues = Object.values(row)
    //         // logger(rowValues)
    //
    //         const parsed: any = this.parseDateTime_(rowValues[0] as string)
    //         const newDateTime = parsed.date
    //
    //         this.dateTime.push(newDateTime)
    //         this.data.push(rowValues.slice(1).map(x => Number(x)))
    //         this.normalizedDayOfYear.push(parsed.normalizedDayOfYear)
    //         this.normalizedTimeOfDay.push(parsed.normalizedTimeOfDay)
    //     })
    //
    //     this.numRows = this.data.length
    //     this.numColumns = this.data[0].length
    //     this.numColumnsExcludingTarget = this.data[0].length - 1
    //
    //     await this.calculateMeansAndStddevs_()
    //
    //     // this.sampleData = {
    //     //     data: tf.tensor2d(this.data, [this.numRows, this.numColumns]),
    //     //     normalizedTimeOfDay: tf.tensor2d(this.normalizedTimeOfDay, [this.numRows, 1])
    //     // }
    //
    //     logger('spend time: ', (performance.now() - beginMs) / 1000)
    //     logger(this.normalizedData)
    // }

    parseDateTime_ = (str: string): IParsedDate => {
        // The date time string with a format that looks like: "17.01.2009 22:10:00"
        const items = str.split(' ')
        const dateStr = items[0]
        const dateStrItems = dateStr.split('.')
        const day = +dateStrItems[0]
        const month = +dateStrItems[1] - 1 // month is 0-based in JS `Date` class.
        const year = +dateStrItems[2]

        const timeStrItems = items[1].split(':')
        const hours = +timeStrItems[0]
        const minutes = +timeStrItems[1]
        const seconds = +timeStrItems[2]

        const date = new Date(Date.UTC(year, month, day, hours, minutes, seconds))
        const yearOnset = new Date(year, 0, 1)
        // normalizedDayOfYear: Day of the year, normalized between 0 and 1.
        const normalizedDayOfYear = (date.getTime() - yearOnset.getTime()) / (366 * 1000 * 60 * 60 * 24)
        const dayOnset = new Date(year, month, day)
        // normalizedTimeOfDay: Time of the day, normalized between 0 and 1.
        const normalizedTimeOfDay = (date.getTime() - dayOnset.getTime()) / (1000 * 60 * 60 * 24)
        return { date, normalizedDayOfYear, normalizedTimeOfDay }
    }

    /**
     * Calculate the means and standard deviations of every column.
     *
     * TensorFlow.js is used for acceleration.
     */
    calculateMeansAndStddevs_ = async (): Promise<void> => {
        tf.tidy(() => {
            // Instead of doing it on all columns at once, we do it
            // column by column, as doing it all at once causes WebGL OOM
            // on some machines.
            this.means = []
            this.stddevs = []
            for (const columnName of this.dataColumnNames) {
                const data = tf.tensor1d(this.getColumnData(columnName).slice(0, 6 * 24 * 365))
                const moments = tf.moments(data)
                this.means.push(moments.mean.dataSync()[0])
                this.stddevs.push(Math.sqrt(moments.variance.dataSync()[0]))
            }
            // console.log('means:', this.means)
            // console.log('stddevs:', this.stddevs)
        })

        // Cache normalized values.
        this.normalizedData = []
        for (let i = 0; i < this.numRows; ++i) {
            const row = []
            for (let j = 0; j < this.numColumns; ++j) {
                row.push((this.data[i][j] - this.means[j]) / this.stddevs[j])
            }
            this.normalizedData.push(row)
        }
    }

    getDataColumnNames = (): string[] => {
        return this.dataColumnNames
    }

    getDataLength = (): number => {
        return this.data.length
    }

    getSampleData = (): tf.TensorContainerObject | undefined => {
        // logger('sampleData', this.sampleData)
        return this.sampleData
    }

    getTime = (index: number): Date => {
        return this.dateTime[index]
    }

    /** Get the mean and standard deviation of a data column. */
    getMeanAndStddev = (dataColumnName: string): any => {
        if (this.means == null || this.stddevs == null) {
            throw new Error('means and stddevs have not been calculated yet.')
        }

        const index = this.getDataColumnNames().indexOf(dataColumnName)
        if (index === -1) {
            throw new Error(`Invalid data column name: ${dataColumnName}`)
        }
        return {
            mean: this.means[index], stddev: this.stddevs[index]
        }
    }

    getColumnData = (columnName: string, includeTime?: boolean, normalize?: boolean,
        beginIndex?: number, length?: number, stride?: number): any[] => {
        const columnIndex = this.dataColumnNames.indexOf(columnName)
        // tf.util.assert(columnIndex >= 0, `Invalid column name: ${columnName}`)

        if (beginIndex == null) {
            beginIndex = 0
        }
        if (length == null) {
            length = this.numRows - beginIndex
        }
        if (stride == null) {
            stride = 1
        }
        const out = []
        for (let i = beginIndex; i < beginIndex + length && i < this.numRows; i += stride) {
            let value: any = normalize ? this.normalizedData[i][columnIndex] : this.data[i][columnIndex]
            if (includeTime) {
                value = { x: this.dateTime[i].getTime(), y: value as number }
            }
            out.push(value)
        }
        return out
    }

    getNextBatchFunction = (shuffle: boolean, lookBack: number, delay: number, batchSize: number, step: number, minIndex: number, maxIndex: number, normalize: boolean,
        includeDateTime: boolean): any => {
        let startIndex = minIndex + lookBack
        const lookBackSlices = Math.floor(lookBack / step)

        return {
            next: () => {
                const rowIndices = []
                let done = false // Indicates whether the dataset has ended.
                if (shuffle) {
                    // If `shuffle` is `true`, start from randomly chosen rows.
                    const range = maxIndex - (minIndex + lookBack)
                    for (let i = 0; i < batchSize; ++i) {
                        const row = minIndex + lookBack + Math.floor(Math.random() * range)
                        rowIndices.push(row)
                    }
                } else {
                    // If `shuffle` is `false`, the starting row indices will be sequential.
                    let r = startIndex
                    for (; r < startIndex + batchSize && r < maxIndex; ++r) {
                        rowIndices.push(r)
                    }
                    if (r >= maxIndex) {
                        done = true
                    }
                }

                const numExamples = rowIndices.length
                startIndex += numExamples

                const featureLength =
                    includeDateTime ? this.numColumns + 2 : this.numColumns
                const samples = tf.buffer([numExamples, lookBackSlices, featureLength])
                const targets = tf.buffer([numExamples, 1])
                // Iterate over examples. Each example contains a number of rows.
                for (let j = 0; j < numExamples; ++j) {
                    const rowIndex = rowIndices[j]
                    let exampleRow = 0
                    // Iterate over rows in the example.
                    for (let r = rowIndex - lookBack; r < rowIndex; r += step) {
                        let exampleCol = 0
                        // Iterate over features in the row.
                        for (let n = 0; n < featureLength; ++n) {
                            let value
                            if (n < this.numColumns) {
                                value = normalize ? this.normalizedData[r][n] : this.data[r][n]
                            } else if (n === this.numColumns) {
                                // Normalized day-of-the-year feature.
                                value = this.normalizedDayOfYear[r]
                            } else {
                                // Normalized time-of-the-day feature.
                                value = this.normalizedTimeOfDay[r]
                            }
                            samples.set(value, j, exampleRow, exampleCol++)
                        }

                        const value = normalize
                            ? this.normalizedData[r + delay][this.tempCol]
                            : this.data[r + delay][this.tempCol]
                        targets.set(value, j, 0)
                        exampleRow++
                    }
                }
                return {
                    value: { xs: samples.toTensor(), ys: targets.toTensor() },
                    done
                }
            }
        }
    }

    dispose = (): void => {
        // todo
    }
}
