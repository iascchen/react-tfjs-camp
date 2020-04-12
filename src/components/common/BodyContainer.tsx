import React from 'react'
import { renderRoutes } from 'react-router-config'

import routes from '../../routers'
import ErrorBoundary from './ErrorBoundary'

const BodyContainer = (): JSX.Element => {
    return (
        <div style={{ padding: 24, background: '#fff', minHeight: '80vh' }}>
            <ErrorBoundary>
                {renderRoutes(routes)}
            </ErrorBoundary>
        </div>
    )
}

export default BodyContainer
