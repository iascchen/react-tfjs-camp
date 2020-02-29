import React, { useState } from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Layout } from 'antd'

import './App.css'
import SideBar from './components/common/SideBar'
import BodyContainer from './components/common/BodyContainer'
import GitHubLogo from './components/common/GitHubLogo'
import ErrorBoundary from './components/common/ErrorBoundary'

const { Header, Sider, Footer } = Layout

const App = (): JSX.Element => {
    const [sCollapsed, setCollapsed] = useState(false)

    const onCollapse = (): void => {
        setCollapsed(collapsed => !collapsed)
    }

    return (
        <Layout>
            <Router>
                <Sider collapsible collapsed={sCollapsed} onCollapse={onCollapse}>
                    <SideBar/>
                </Sider>
                <Layout className='site-layout'>
                    <Header style={{ background: '#fff', padding: '0' }}>
                        <span style={{ margin: '0 8px' }}>Tensorflow.js React-Hooks Playground</span>
                        <GitHubLogo/>
                    </Header>
                    <ErrorBoundary>
                        <BodyContainer/>
                    </ErrorBoundary>
                    <Footer style={{ textAlign: 'center' }}>©2020 Created by Iasc CHEN(iascchen@gmail.com)</Footer>
                </Layout>
            </Router>
        </Layout>
    )
}

export default App
