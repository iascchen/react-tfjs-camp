import { RouteConfig } from 'react-router-config'

import Home from './components/common/Home'

import Curve from './components/curve/Curve'
import Iris from './components/iris/Iris'

import MnistLayersApiImpl from './components/mnist/MnistLayersApiImpl'
import MnistCoreApiImpl from './components/mnist/MnistCoreApiImpl'

import MobileNetClassifier from './components/mobilenet/MobileNetClassifier'
import MobileNetKnnClassifier from './components/mobilenet/MobileNetKnnClassifier'
import MobileNetTransfer from './components/mobilenet/MobileNetTransfer'
import MobileNetObjDetector from './components/mobilenet/MobileNetObjDetector'

import JenaWeather from './components/rnn/JenaWeather'
import SentimentImdb from './components/rnn/SentimentImdb'
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

    { path: '/mnist/layers', component: MnistLayersApiImpl },
    { path: '/mnist/core', component: MnistCoreApiImpl },

    { path: '/mobilenet/basic', component: MobileNetClassifier },
    { path: '/mobilenet/knn', component: MobileNetKnnClassifier },
    { path: '/mobilenet/transfer', component: MobileNetTransfer },
    { path: '/mobilenet/objdetector', component: MobileNetObjDetector },

    { path: '/rnn/jena', component: JenaWeather },
    { path: '/rnn/sentiment', component: SentimentImdb },
    { path: '/rnn/lstm', component: TextGenLstm },

    { path: '/pretrained/handpose', component: HandPosePanel },
    { path: '/pretrained/facemesh', component: FaceMeshPanel },
    { path: '/pretrained/posenet', component: PoseNetPanel },

    { path: '/sandbox/fetch', component: FetchWidget },
    { path: '/sandbox/array', component: TypedArrayWidget },
    { path: '/sandbox/tfvis', component: TfvisWidget },

    { path: '*', component: Home }
]

export default routes
