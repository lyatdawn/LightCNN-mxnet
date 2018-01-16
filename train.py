# -*- coding:utf-8 -*-
"""
Train LightCNN-9 and LightCNN-29.
"""
import os
import argparse
import logging
import mxnet as mx
from common import fit, data
from symbols import light_cnn

# logging
log_file = "./model/LightCNN.log"
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                   filename=log_file,
                   level=logging.INFO,
                   filemode='a+')
logging.getLogger().addHandler(logging.StreamHandler())

'''
# data load method one. define function to load data.
def get_rec_iter(args, kv):
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, "MsCeleb_train.rec"),
        data_shape  = (1, 128, 128),
        scale       = 1./255,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, "MsCeleb_val.rec"),
        batch_size  = args.batch_size,
        data_shape  = (1, 128, 128),
        scale       = 1./255,
        rand_crop   = True,
        rand_mirror = False,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return train, val
'''

# data load method two. use common/data.py dataget_rec_iter() to load ImageRecordIter.
def load_rec_iter():
    # data_dir = "/home/ly/DATASETS/CASIA", # CASIA
    data_dir = "/home/ly/DATASETS/MsCelebV1/MXNet_MsCeleb_Faces_Aligned" # MsCeleb
    # REC data is 144*144 image, use random crop to 128*128.
    fnames = (os.path.join(data_dir, "MsCeleb_train.rec"), os.path.join(data_dir, "MsCeleb_val.rec"))
    
    return fnames

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train LightCNN-9 and LightCNN-29.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use_net", type=str, default="LightCNN-29",
                        help='which net will be used.')

    # load rec data
    (train_fname, val_fname) = load_rec_iter()

    fit.add_fit_args(parser)
    # If use common/data.py dataget_rec_iter() to load ImageRecordIter, can refer to train_imagenet.py. 
    # RGB/gray image. use get_rec_iter() to load ImageRecordIter, can appoint some augmentations.
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # data.set_data_aug_level(parser, 2) # not need
    # or, we can refer to mxnet-face/verification/lightened_cnn.py to train/val.
    parser.set_defaults(
        # network
        network = "light_cnn", # use network to appoint the init method
        # data
        # num_classes = 10575, # CASIA
        num_classes = 10000, # MsCeleb
        # num_examples = 393512, # CASIA
        num_examples = 562461, # MsCeleb
        # data augmentations
        data_train = train_fname,
        data_val = val_fname,
        image_shape = '1,128,128', # image shape
        pad_size = 4,
        data_nthreads = 8, # number of threads for data decoding
        # train
        gpus = '0',
        batch_size = 128,
        disp_batches = 100,
        num_epochs = 55, # num_epochs >= load_epoch
        # optim
        lr = 0.0001,
        lr_step_epochs = '0,100', # lr is 1e-4, lr_step_epochs='0', lr_factor=0.1, so the init lr is 1e-5.
        # larger than 50 not use, so in the training. Optim algorithm is ADAM with initial lr is 1e-5.
        optimizer = 'adam', # Use ADAM
        
        # save checkpoint
        load_epoch = 50, # auto load pre trained model, in fit.py, use _load_model() funtion to load pre trained
        # From random init to train.

        # chechpoint
        model_prefix = 'model/LightCNN',
    )
    args = parser.parse_args()

    # load network
    if args.use_net == "LightCNN-9":
        sym = light_cnn.LightCNN_9(name="lightcnn_9", num_classes=args.num_classes)
    else:
        sym = light_cnn.LightCNN_29(name="lightcnn_29", num_classes=args.num_classes)

    # train
    fit.fit(args, sym, data.get_rec_iter) 
    # Modify common/fit.py the init method in line 181.

    # In train_helper.py, do some test, to checkup weight and bias init is right or not. Use common/fit.py to
    # train/val.
    # In this way, coule do some augmentations for data; otherwise, data augmentations will need define by self.
    