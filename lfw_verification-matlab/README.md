利用自己编写的LightCNN-29, 在LFW数据集上进行人脸验证, 结果如下:
- run lfw_eva.m to get the accuracy
ACC: 0.857500 
EER: 0.149333 
AUC: 0.922906
1) 在mxnet中, 提取一张图像的特征时, 网络输入应该不用/255., mxnet官网上的predict程序, 也没有/255..