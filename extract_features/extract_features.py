# -*- coding:utf-8 -*-
"""
Valing LightCNN.
"""
import argparse
import mxnet as mx
import cv2
import numpy as np
import scipy.io as sio

def get_image(image_path):
    # download and show the image
    img = cv2.imread(image_path, 0) # gray
    if img is None:
        return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (128, 128))
    img = np.reshape(img, (1, 1, 128, 128))
    # TODO: In mxnet, if we want to extract a image's feature, the input of network don't need to divide 255.?
    # Maybe...

    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-prefix', default='../model/LightCNN',
                        help='The trained model to get feature')
    parser.add_argument('--load-epoch', type=int, default=50,
                        help='The epoch number of model')
    args = parser.parse_args()

    # Refer to https://www.2cto.com/kf/201706/646130.html to use pretrained model.
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)

    internals = sym.get_internals() # Use above sym! 
    # print(internals.list_outputs()[-10:]) # use list_outputs() to get print.
    fe_sym = internals["lightcnn_29_fc1_output"] # add _output. fc1 output as every image's feature.

    mod = mx.mod.Module(symbol=fe_sym, context=mx.gpu(0)) # Softmax
    mod.bind(for_training=False, data_shapes=[('data', (1, 1, 128, 128))], 
             label_shapes=mod._label_shapes)
    # image still is 128*128.
    mod.set_params(arg_params, aux_params)

    # Dataset is lfw_patch, total have 13233 images. Extract every image's feture, the feature dim is 768.
    labels = np.empty([13233, 1], dtype=object)
    res = []
    count = 0
    with open("lfw_patch_part.txt", "r") as f:
        # lfw_patch_part.txt
        # 13233 samples.
        for line in f:
            name = []
            line = line.strip() # get rid of ' '.
            name.append(line.split('/')[-2] + '/' + line.split('/')[-1])
            # print(name) # Aaron_Peirsol/Aaron_Peirsol_0004.jpg
            labels[count, 0] = name

            image_path = "/home/ly/DATASETS" + line

            img = get_image(image_path)
            # compute the predict probabilities
            mod.forward(mx.io.DataBatch([mx.nd.array(img)]), is_train=False)
            feature = mod.get_outputs()[0].asnumpy()
            feature = np.squeeze(feature)
            # print(feature.shape) # 768.
            res.append(feature)
            count += 1
            if count % 100 == 0:
                print("Images {}".format(count))

    res = np.array(res)
    res = np.reshape(res, [13233, 768])
    print (res.shape)
    print (labels.shape)
    sio.savemat("./LFW_features.mat", {'data':res, 'label':labels})
    f.close()