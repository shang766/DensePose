# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""VGG16 from https://arxiv.org/abs/1409.1556."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg

##VGG16 13层卷积，3个全连接层。根据4个MaxPool分成了5个部分。
def add_VGG16_conv5_body(model):
    model.Conv('data', 'conv1_1', 3, 64, 3, pad=1, stride=1) #Conv(输入数据名称，输出数据名称，输入的通道个数，输出的通道个数，卷积核尺寸（3*3），pad=1（same）,stride 步长为1）
    model.Relu('conv1_1', 'conv1_1')#激活函数
    model.Conv('conv1_1', 'conv1_2', 64, 64, 3, pad=1, stride=1)#卷积第二次
    model.Relu('conv1_2', 'conv1_2')
    model.MaxPool('conv1_2', 'pool1', kernel=2, pad=0, stride=2)  # filter 2*2*64 且stride 为2 尺寸减半 
    model.Conv('pool1', 'conv2_1', 64, 128, 3, pad=1, stride=1)   #通道数从64 变到128
    model.Relu('conv2_1', 'conv2_1')
    model.Conv('conv2_1', 'conv2_2', 128, 128, 3, pad=1, stride=1)#卷积第二次
    model.Relu('conv2_2', 'conv2_2')
    model.MaxPool('conv2_2', 'pool2', kernel=2, pad=0, stride=2) # filter 2*2*64 且stride 为2 尺寸减半
    model.StopGradient('pool2', 'pool2')
    model.Conv('pool2', 'conv3_1', 128, 256, 3, pad=1, stride=1)#通道数从128 变到256
    model.Relu('conv3_1', 'conv3_1')
    model.Conv('conv3_1', 'conv3_2', 256, 256, 3, pad=1, stride=1)#卷积第二次
    model.Relu('conv3_2', 'conv3_2')
    model.Conv('conv3_2', 'conv3_3', 256, 256, 3, pad=1, stride=1)#卷积第三次
    model.Relu('conv3_3', 'conv3_3')
    model.MaxPool('conv3_3', 'pool3', kernel=2, pad=0, stride=2) # filter 2*2*64 且stride 为2 尺寸减半
    model.Conv('pool3', 'conv4_1', 256, 512, 3, pad=1, stride=1)#通道数从256 变到512
    model.Relu('conv4_1', 'conv4_1')
    model.Conv('conv4_1', 'conv4_2', 512, 512, 3, pad=1, stride=1)#卷积第二次
    model.Relu('conv4_2', 'conv4_2')
    model.Conv('conv4_2', 'conv4_3', 512, 512, 3, pad=1, stride=1)#卷积第三次
    model.Relu('conv4_3', 'conv4_3')
    model.MaxPool('conv4_3', 'pool4', kernel=2, pad=0, stride=2) # filter 2*2*64 且stride 为2 尺寸减半
    model.Conv('pool4', 'conv5_1', 512, 512, 3, pad=1, stride=1)#通道数还是512
    model.Relu('conv5_1', 'conv5_1')
    model.Conv('conv5_1', 'conv5_2', 512, 512, 3, pad=1, stride=1)#卷积第二次
    model.Relu('conv5_2', 'conv5_2')
    model.Conv('conv5_2', 'conv5_3', 512, 512, 3, pad=1, stride=1)#卷积第三次
    blob_out = model.Relu('conv5_3', 'conv5_3')
    return blob_out, 512, 1. / 16.


def add_VGG16_roi_fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=7,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC('pool5', 'fc6', dim_in * 7 * 7, 4096)##全连接层（输入名称，输出名称，输入的节点，输出的节点）
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', 4096, 4096)
    blob_out = model.Relu('fc7', 'fc7')
    return blob_out, 4096
