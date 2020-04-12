import React from 'react'
import { renderRoutes } from 'react-router-config'
import { Alert } from 'antd'

import routes from '../../routers'

const { ErrorBoundary } = Alert

const BodyContainer = (): JSX.Element => {
    return (
        <div style={{ padding: 24, background: '#ffffff', minHeight: '80vh' }}>
            <ErrorBoundary>
                {renderRoutes(routes)}
            </ErrorBoundary>
        </div>
    )
}

export default BodyContainer
