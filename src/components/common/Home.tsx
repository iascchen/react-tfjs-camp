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
            <h1>Learn React Hooks and Tensorflow.js</h1>
            <p>Used other open source resources</p>
            <ul>
                <li>Ant.Design</li>
                <li>Ant.V Bizchart</li>
                <li>Teachable Machine</li>
                <li>Posenet</li>
            </ul>
        </div>
    )
}

export default Home
