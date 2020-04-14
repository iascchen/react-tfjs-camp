import React from 'react'
import { Col, Row } from 'antd'

import reactLogo from '../../react_logo.svg'
import tsLogo from '../../typescript_logo.svg'
const tfLogo = '/images/tf_logo.jpeg'

const Home = (): JSX.Element => {
    return (
        <>
            <h1>RTCamp: React Tensorflow.js Camp</h1>
            <header className='App-header'>
                <Row>
                    <Col span={8} className='centerHeader'>
                        <img src={reactLogo} alt='logo' style={{ width: '90%' }}/>
                    </Col>
                    <Col span={8} className='centerHeader'>
                        <img src={tfLogo} alt='logo' style={{ width: '100%' }}/>
                    </Col>
                    <Col span={8} className='centerHeader'>
                        <img src={tsLogo} alt='logo' style={{ width: '90%' }}/>
                    </Col>
                </Row>
            </header>

            <h2>Tensorflow.js Study Camp -- full-stack develop from zero</h2>
            <h3>Powered by: </h3>
            <ul>
                <li>Tensorflow.js 1.7.1, tfjs-example and tfjs-model</li>
                <li>React 16.13 (React Hooks)</li>
                <li>Typescript 3.7.2</li>
                <li>Node.js / ts-node</li>
            </ul>
            <ul>
                <li>Ant.Design v4.1.3</li>
                <li>Ant.V Bizchart </li>
                <li>Teachable Machine</li>
            </ul>
        </>
    )
}

export default Home
