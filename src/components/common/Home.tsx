import React from 'react'
import { Col, Row } from 'antd'

import reactLogo from '../../react_logo.svg'
import tsLogo from '../../typescript_logo.svg'

const tfLogo = '/tf_logo.jpeg'

const Home = (): JSX.Element => {
    return (
        <div className='App'>
            <header className='App-header'>
                <Row>
                    <Col span={8}><img src={reactLogo} height={200} alt='logo'/></Col>
                    <Col span={8}><img src={tfLogo} height={200} alt='logo2'/></Col>
                    <Col span={8}><img src={tsLogo} height={100} alt='logo3'/></Col>
                </Row>
            </header>
            <p>Learn React Hooks and Tensorflow.js</p>
        </div>
    )
}

export default Home
