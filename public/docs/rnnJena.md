## RNN 文本生成：自然语言处理

### Jena

RNN

RNN 比 GRU 和 LSTM 都更容易理解。

RNN 的伪代码

        y=0
        for x in input_sequence:
            y = f(dot(W, x) + dot(U, y))

GRU 的伪代码

        h=0
        for x_i in input_sequence:
            z = sigmoid(dot(W_z, x) + dot(U_z, h))
            r = sigmoid(dot(W_r, x) + dot(W_r, h)) 
            h_prime = tanh(dot(W, x) + dot(r, dot(U, h))) 
            h = dot(1 - z, h) + dot(z, h_prime)

LSTM


### 使用 RNN 进行语义分析

对文章进行 multi-hot 分析，对 IMDB 数据进行分析。

更高效的技术：词嵌入



我们已经了解了如何使用和构建自己的 CNN 网络。
接下来，我们来认识另外一个重要的深度神经网络模型：循环神经网络 RNN
RNN 被用于处理文本理解、语音识别等场景。这类场景的共同特点是：

**问题结果与过去的几个输入数据都相关**

简单说来就是 y = f( Wx + Uy)



RNN



LSTM

中文分词

作诗、作文、作曲Magenta

生成模型：图像风格迁移，

seq-to-seq 的其他应用，DNA 碱基对，代码生成。。。

BERT

使用 RNN 处理视频、音频。。。

## 第九章 声音的处理：对信号的AI处理

傅立叶变换将信号转成频谱图 -> 图像处理

声音的输入

基于 LSTM 的 语音识别

## GAN 对抗生成网络
