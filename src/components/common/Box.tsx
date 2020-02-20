import React, { useState } from 'react'

interface IProps {
    children: JSX.Element
}
const Box = (props: IProps): JSX.Element => {
    const [d] = useState(1)
    console.log(d)

    return (
        <div>
            {props.children}
        </div>
    )
}

export default Box
