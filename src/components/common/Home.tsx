import React from 'react'
import { Col, Row } from 'antd'

import reactLogo from '../../react_logo.svg'
import tsLogo from '../../typescript_logo.svg'

const tfLogo = '/tf_logo.jpeg'

const Home = (): JSX.Element => {
    return (
        <div>
            <header className='App-header'>
                <Row>
                    <Col span={8}><img src={reactLogo} height={200} alt='logo'/></Col>
                    <Col span={8}><img src={tfLogo} height={200} alt='logo2'/></Col>
                    <Col span={8}><img src={tsLogo} height={100} alt='logo3'/></Col>
                </Row>
            </header>
            <h2>RTP: React Tensorflow.js Playground</h2>
            <h2>A Tensorflow.js study tool, coding with React Hooks and Typescript</h2>
            <p>Used other open source resources</p>
            <ul>
                <li>Tensorflow.js 1.5.2</li>
                <li>React 16.12 (React Hooks)</li>
                <li>Typescript 3.7.2</li>
            </ul>
            <ul>
                <li>Ant.Design v4.0.0</li>
                <li>Ant.V Bizchart</li>
                <li>Teachable Machine</li>
            </ul>
        </div>
    )
}

export default Home
