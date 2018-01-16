# -*- coding:utf-8 -*-
"""
LightCNN-9, LightCNN-29 network.
Equation (4) element-wise max: max(x1, x2, x3). The num_filter might * 3/2.

1. Conv operations size: [(m + 2 * padding - kernel_size) / stride] + 1.
2. mx.symbol.maximum() has only two arguments. So, can use a aux variable.
"""
import mxnet as mx

def mfm(name, data, num_filter, kernel_size, stride, padding, type=1):
    # type=1, conv + ele-max + conv + ele-max(resblock); otherwise, conv + ele-max.
    # MFM is the element-wise max.
    if type==1:
        conv_1 = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel_size, stride=stride, 
            pad=padding, name=name + "_conv_1")
        slice_conv1 = mx.symbol.SliceChannel(data=conv_1, num_outputs=3, name=name + "_slice_conv1")
        aux_mfm_conv1 = mx.symbol.maximum(slice_conv1[0], slice_conv1[1])
        mfm_conv1 = mx.symbol.maximum(aux_mfm_conv1, slice_conv1[2])

        conv = mx.symbol.Convolution(data=mfm_conv1, num_filter=num_filter, kernel=kernel_size, stride=stride, 
            pad=padding, name=name + "_conv")
    else:
        conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel_size, stride=stride, 
            pad=padding, name=name + "_conv")

    slice_conv = mx.symbol.SliceChannel(data=conv, num_outputs=3, name=name + "_slice_conv")
    aux_slice_conv = mx.symbol.maximum(slice_conv[0], slice_conv[1])
    mfm_conv = mx.symbol.maximum(aux_slice_conv, slice_conv[2])

    return mfm_conv

def LightCNN_9(name="lightcnn_9", num_classes=10575):
    # name="lightcnn_9"
    data = mx.symbol.Variable(name="data")
    # data shape: 128 * 128.
    # mfm: type=0, conv + ele-max.
    mfm_1 = mfm(name=name + "_mfm_1", data=data, num_filter=144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), 
        type=0)
    # mfm_1 shape: 128 * 128 --> conv out: 128 * 128 * 144 --> ele-max out: 128 * 128 * 48.
    pool1 = mx.symbol.Pooling(data=mfm_1, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool1")
    # pool1 shape: 128 * 128 * 48 --> pool out: 64 * 64 * 48.
    
    mfm_2a = mfm(name=name + "_mfm_2a", data=pool1, num_filter=144, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), 
        type=0)
    # mfm_2a shape: 64 * 64 * 128 --> conv out: 64 * 64 * 144 --> ele-max out: 64 * 64 * 48.
    mfm_2 = mfm(name=name + "_mfm_2", data=mfm_2a, num_filter=288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_2 shape: 64 * 64 * 128 --> conv out: 64 * 64 * 288 --> ele-max out: 64 * 64 * 96.
    pool2 = mx.symbol.Pooling(data=mfm_2, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool2")
    # pool2 shape: 64 * 64 * 96 --> pool out: 32 * 32 * 96.
    
    mfm_3a = mfm(name=name + "_mfm_3a", data=pool2, num_filter=288, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), 
        type=0)
    # mfm_3a shape: 32 * 32 * 96 --> conv out: 32 * 32 * 288 --> ele-max out: 32 * 32 * 96.
    mfm_3 = mfm(name=name + "_mfm_3", data=mfm_3a, num_filter=576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_3 shape: 32 * 32 * 96 --> conv out: 32 * 32 * 576 --> ele-max out: 32 * 32 * 192.
    pool3 = mx.symbol.Pooling(data=mfm_3, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool3")
    # pool3 shape: 32 * 32 * 192 --> pool out: 16 * 16 * 192.
    
    mfm_4a = mfm(name=name + "_mfm_4a", data=pool3, num_filter=576, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), 
        type=0)
    # mfm_4a shape: 16 * 16 * 192 --> conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192.
    mfm_4 = mfm(name=name + "_mfm_4", data=mfm_4a, num_filter=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_4 shape: 16 * 16 * 192 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128.
    
    mfm_5a = mfm(name=name + "_mfm_5a", data=mfm_4, num_filter=384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), 
        type=0)
    # mfm_5a shape: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128.
    mfm_5 = mfm(name=name + "_mfm_5", data=mfm_5a, num_filter=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_5 shape: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128.
    pool4 = mx.symbol.Pooling(data=mfm_5, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool4")
    # pool4 shape: 16 * 16 * 128 --> pool out: 8 * 8 * 128.

    flatten = mx.symbol.Flatten(data=pool4)
    # flatten shape: 8192.
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=768, name=name + "_fc1")
    # fc1 shape: 768.
    slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=3, name=name + "_slice_fc1")
    aux_mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
    mfm_fc1 = mx.symbol.maximum(aux_mfm_fc1, slice_fc1[2])
    # mfm_fc1 shape: 256.

    fc2 = mx.symbol.FullyConnected(data=mfm_fc1, num_hidden=num_classes, name=name +"_fc2")
    # fc2 shape: 10575.
    softmax_9 = mx.symbol.SoftmaxOutput(data=fc2, name='softmax') # name='softmax'
    return softmax_9

def resblock(name, data, num_blocks, num_filter, kernel_size, stride, padding):
    '''
    residual blocks contain two 3x3 convolution layers, and two MFM operations without batch normalization.
    resblock contain num_blocks operations. num_blocks = [1, 2, 3, 4].
    As follow:
        x
        |\
        | \
        |  conv2d + mfm
        |  conv2d + mfm
        | /
        |/
        + (addition here)
        |
      out
    '''
    for i in range(0, num_blocks):
        mfm_x = mfm(name=name, data=data, num_filter=num_filter, kernel_size=kernel_size, stride=stride, 
            padding=padding, type=1)
    out = mfm_x + data
    return out


def LightCNN_29(name="lightcnn_29", num_classes=10575):
    # name="lightcnn_29". Void duplicate; The num_filter might * 3/2; element-wise max: max(x1, x2, x3).
    data = mx.symbol.Variable(name="data")
    # data shape: 128 * 128.
    num_blocks = [1, 2, 3, 4]

    mfm_1 = mfm(name=name + "_mfm_1", data=data, num_filter=144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), 
        type=0)
    # mfm_1 shape: 128 * 128 --> conv out: 128 * 128 * 144 --> ele-max out: 128 * 128 * 48.
    pool1 = mx.symbol.Pooling(data=mfm_1, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool1")
    # pool1 shape: 128 * 128 * 48 --> pool out: 64 * 64 * 48.

    # 1st residual blocks.
    mfm_2x = resblock(name=name + "_mfm_2x", data=pool1, num_blocks=num_blocks[0], num_filter=144, kernel_size=(3, 3), 
        stride=(1, 1), padding=(1, 1))
    # mfm_2x shape: 64 * 64 * 48 --> conv out: 64 * 64 * 144 --> ele-max out: 64 * 64 * 48 -->
    # conv out: 64 * 64 * 144 --> ele-max out: 64 * 64 * 48.
    mfm_2a = mfm(name=name + "_mfm_2a", data=mfm_2x, num_filter=144, kernel_size=(1, 1), stride=(1, 1), 
        padding=(0, 0), type=0)
    # mfm_2a shape: 64 * 64 *48 --> conv out: 64 * 64 * 144 --> ele-max out: 64 * 64 * 48.
    mfm_2 = mfm(name=name + "_mfm_2", data=mfm_2a, num_filter=288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_2 shape: 64 * 64 *48 --> conv out: 64 * 64 * 288 --> ele-max out: 64 * 64 * 96.
    pool2 = mx.symbol.Pooling(data=mfm_2, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool2")
    # pool2 shape: 64 * 64 * 96 --> pool out: 32 * 32 * 96.

    # 2nd residual blocks.
    mfm_3x = resblock(name=name + "_mfm_3x", data=pool2, num_blocks=num_blocks[1], num_filter=288, kernel_size=(3, 3), 
        stride=(1, 1), padding=(1, 1))
    # mfm_3x shape:  32 * 32 * 96 --> conv out: 32 * 32 * 288 --> ele-max out: 32 * 32 * 96 -->
    # conv out: 32 * 32 * 288 --> ele-max out: 32 * 32 * 96 --> conv out: 32 * 32 * 288 --> ele-max out: 32 * 32 * 96 -->
    # conv out: 32 * 32 * 288 --> ele-max out: 32 * 32 * 96.
    mfm_3a = mfm(name=name + "_mfm_3a", data=mfm_3x, num_filter=288, kernel_size=(1, 1), stride=(1, 1), 
        padding=(0, 0), type=0)
    # mfm_3a shape: 32 * 32 * 96 --> conv out: 32 * 32 * 288 --> ele-max out: 32 * 32 * 96.
    mfm_3 = mfm(name=name + "_mfm_3", data=mfm_3a, num_filter=576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_3 shape: 32 * 32 * 96 --> conv out: 32 * 32 * 576 --> ele-max out: 32 * 32 * 192.
    pool3 = mx.symbol.Pooling(data=mfm_3, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool3")
    # pool3 shape: 32 * 32 * 192 --> pool out: 16 * 16 * 192.

    # 3rd residual blocks.
    mfm_4x = resblock(name=name + "_mfm_4x", data=pool3, num_blocks=num_blocks[2], num_filter=576, kernel_size=(3, 3), 
        stride=(1, 1), padding=(1, 1))
    # mfm_4x shape: 16 * 16 * 192 --> conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192 -->
    # conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192 --> conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192 -->
    # conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192 --> conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192 -->
    # conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192.
    mfm_4a = mfm(name=name + "_mfm_4a", data=mfm_4x, num_filter=576, kernel_size=(1, 1), stride=(1, 1), 
        padding=(0, 0), type=0)
    # mfm_4a shape: 16 * 16 * 192 --> conv out: 16 * 16 * 576 --> ele-max out: 16 * 16 * 192.
    mfm_4 = mfm(name=name + "_mfm_4", data=mfm_4a, num_filter=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_4 shape: 16 * 16 * 192 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128.

    # 4st residual blocks.
    mfm_5x = resblock(name=name + "_mfm_5x", data=mfm_4, num_blocks=num_blocks[3], num_filter=384, kernel_size=(3, 3), 
        stride=(1, 1), padding=(1, 1))
    # mfm_5x shape: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128 -->
    # conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128 -->
    # conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128 -->
    # conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128 -->
    # conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128.
    mfm_5a = mfm(name=name + "_mfm_5a", data=mfm_5x, num_filter=384, kernel_size=(1, 1), stride=(1, 1), 
        padding=(0, 0), type=0)
    # mfm_5a shape: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128.
    mfm_5 = mfm(name=name + "_mfm_5", data=mfm_5a, num_filter=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
        type=0)
    # mfm_5 shape: 16 * 16 * 128 --> conv out: 16 * 16 * 384 --> ele-max out: 16 * 16 * 128.
    pool4 = mx.symbol.Pooling(data=mfm_5, kernel=(2, 2), stride=(2,2), pool_type="max", name=name + "_pool4")
    # pool4 shape: 16 * 16 * 128 --> pool out: 8 * 8 * 128.

    flatten = mx.symbol.Flatten(data=pool4)
    # flatten shape: 8192.
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=768, name=name + "_fc1")
    # fc1 shape: 768.
    slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=3, name=name + "_slice_fc1")
    aux_mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
    mfm_fc1 = mx.symbol.maximum(aux_mfm_fc1, slice_fc1[2])
    # mfm_fc1 shape: 256.

    fc2 = mx.symbol.FullyConnected(data=mfm_fc1, num_hidden=num_classes, name=name +"_fc2")
    # fc2 shape: 10575.
    softmax_29 = mx.symbol.SoftmaxOutput(data=fc2, name='softmax') # name='softmax'
    return softmax_29
