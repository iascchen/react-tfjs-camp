import React, { Component } from 'react'

interface IErrorBoundaryProps {
    children: JSX.Element | React.ReactNode
}

interface IErrorBoundaryState {
    error: Error | null
    errorInfo: Record<string, any> | null
}

export default class ErrorBoundary extends Component<IErrorBoundaryProps, IErrorBoundaryState> {
    constructor (props: IErrorBoundaryProps) {
        super(props)
        this.state = {
            error: null,
            errorInfo: null
        }
    }

    componentDidCatch (error: Error, errorInfo: Record<string, any>): void {
        this.setState({
            error: error,
            errorInfo: errorInfo
        })
    }

    render (): JSX.Element | React.ReactNode {
        const { children } = this.props
        const { error, errorInfo } = this.state

        if (error) {
            // Error path
            return (
                <div>
                    <h2>Something went wrong.</h2>
                    <details style={{ whiteSpace: 'pre-wrap' }}>
                        {error.toString()}
                        <br />
                        {errorInfo?.componentStack}
                    </details>
                </div>
            )
        }

        // return the children
        return children
    }
}
