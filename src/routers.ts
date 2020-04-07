import { RouteConfig } from 'react-router-config'

import Home from './components/common/Home'
import NotFound from './components/common/NotFound'

import Curve from './components/curve/Curve'
import Iris from './components/iris/Iris'

import MnistKeras from './components/mnist/MnistKeras'
import MnistCore from './components/mnist/MnistCore'

import MobilenetClassifier from './components/mobilenet/MobilenetClassifier'
import MobilenetKnnClassifier from './components/mobilenet/MobilenetKnnClassifier'
import MobilenetTransfer from './components/mobilenet/MobilenetTransfer'
import MobilenetObjDetector from './components/mobilenet/MobilenetObjDetector'

import JenaWeather from './components/rnn/JenaWeather'
import ImdbSentiment from './components/rnn/ImdbSentiment'
import TextGenLstm from './components/rnn/TextGenLstm'

import HandPosePanel from './components/pretrained/HandPosePanel'
import FaceMeshPanel from './components/pretrained/FaceMeshPanel'
import PoseNetPanel from './components/pretrained/PoseNetPanel'

import FetchWidget from './components/sandbox/FetchWidget'
import TypedArrayWidget from './components/sandbox/TypedArrayWidget'
import TfvisWidget from './components/sandbox/TfvisWidget'

const routes: RouteConfig[] = [
    { path: '/', exact: true, component: Home },

    { path: '/curve', component: Curve },
    { path: '/iris', component: Iris },

    { path: '/mnist/keras', component: MnistKeras },
    { path: '/mnist/core', component: MnistCore },

    { path: '/mobilenet/basic', component: MobilenetClassifier },
    { path: '/mobilenet/knn', component: MobilenetKnnClassifier },
    { path: '/mobilenet/transfer', component: MobilenetTransfer },
    { path: '/mobilenet/objdetector', component: MobilenetObjDetector },

    { path: '/rnn/jena', component: JenaWeather },
    { path: '/rnn/sentiment', component: ImdbSentiment },
    { path: '/rnn/lstm', component: TextGenLstm },

    { path: '/pretrained/handpose', component: HandPosePanel },
    { path: '/pretrained/facemesh', component: FaceMeshPanel },
    { path: '/pretrained/posenet', component: PoseNetPanel },

    { path: '/sandbox/fetch', component: FetchWidget },
    { path: '/sandbox/array', component: TypedArrayWidget },
    { path: '/sandbox/tfvis', component: TfvisWidget },

    { path: '*', component: NotFound }
]

interface IBreadcrumbMap {
    [index: string]: string
}

export const breadcrumbNameMap: IBreadcrumbMap = {
    '/': 'Home',
    '/curve': 'Curve',
    '/iris': 'Iris',

    '/mnist/keras': 'MNIST Layer Model',
    '/mnist/core': 'MNIST Core API',

    '/mobilenet/basic': 'Mobilenet Classifier',
    '/mobilenet/knn': 'Teachable Machine',
    '/mobilenet/transfer': 'Transfer Learning: Classifier',
    '/mobilenet/objdetector': 'Transfer Learning: Object Detector',

    '/rnn/jena': 'Jena Weather',
    '/rnn/sentiment': 'Sentiment',
    '/rnn/lstm': 'Lstm Text',

    '/pretrained/handpose': 'Hand Pose',
    '/pretrained/facemesh': 'Face Mesh',
    '/pretrained/posenet': 'Pose',

    '/sandbox/array': 'TypedArray',
    '/sandbox/fetch': 'Fetch',
    '/sandbox/tfvis': 'TfVis Widget'
}

export default routes
