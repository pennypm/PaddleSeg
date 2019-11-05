# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from models.backbone.resnet import ResNet as resnet_backbone

def conv_layer(input,
                  num_filters,
                  filter_size=1,
                  stride=1,
                  padding=0,
                  dilation=1,
                  param_attr=None,
                  act=None,
                  name=None):
    
    if param_attr is None:
        param_attr = ParamAttr(name=name + "_weights")
    
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        act=act,
        param_attr=param_attr,
        bias_attr=ParamAttr(name=name + "_biases"),
        name=name + '.conv2d.output')

    return conv


def bn_layer(input, act=None, name=None):
    return fluid.layers.batch_norm(
                 input=input,
                 act=act,
                 name=name + '.output.1',
                 param_attr=ParamAttr(name=name + '_scale'),
                 bias_attr=ParamAttr(name + '_offset'),
                 moving_mean_name=name + '_mean',
                 moving_variance_name=name + '_variance', )

def bn_leaky_relu(input, alpha, name=None):
    bn_data = bn_layer(input, name=name) 
    return fluid.layers.leaky_relu(bn_data, alpha=alpha)


def conv_bn_relu(input, num_filters,
                  filter_size=1,
                  stride=1,
                  padding=0,
                  dilation=1,
                  bias_attr=None,
                  act=None,
                  alpha=0.01,
                  name=None):

    data = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=bias_attr,
        name=name + '.conv2d.output')

    bn_name = name + '_bn'
    if alpha:
        return bn_leaky_relu(data, alpha=alpha, name=bn_name)
    else:
        return bn_layer(data, act=act, name=bn_name)

def psp_module(input, out_features):
    # Pyramid Scene Parsing 金字塔池化模块
    # 输入：backbone输出的特征
    # 输出：对输入进行不同尺度pooling, 卷积操作后插值回原始尺寸，并concat
    #       最后进行一个卷积及BN操作
    
    cat_layers = []
    sizes = (1,2,3,6)
    shape = fluid.layers.shape(input)[2:]
    for size in sizes:
        psp_name = "psp" + str(size)+'_'
        pool = fluid.layers.adaptive_pool2d(input, 
                pool_size=[size, size], 
                pool_type='avg', 
                name=psp_name+'adapool')
        data = conv_bn_relu(pool, 
                out_features, 
                filter_size=1,
                name= psp_name + 'conv')
        interp = fluid.layers.resize_bilinear(data, 
                out_shape=input.shape[2:], 
                actual_shape=shape,
                name=psp_name+'interp') 
        shape.stop_gradient = True
        cat_layers.append(interp)
    cat_layers = cat_layers + [input]
    cat = fluid.layers.concat(cat_layers, axis=1, name='psp_cat')
    out = conv_bn_relu(cat, 
                    out_features, 
                    filter_size=3,
                    padding=1, 
                    name="psp_end")

    return out

def resnet(input):
    # PSPNET backbone: resnet, 默认resnet50
    # end_points: resnet终止层数
    # dilation_dict: resnet block数及对应的膨胀卷积尺度
    scale = 1 
    layers = 101 
    end_points = layers - 1
    decode_points = [10, 22, 91]
    dilation_dict = {3:2}
    model = resnet_backbone(layers, scale, stem='pspnet')
    data, decode_shortcuts = model.net(input, 
                        end_points=end_points, 
                        decode_points=decode_points,
                        dilation_dict=dilation_dict)
    
    decode_shortcuts = [decode_shortcuts[i] for i in decode_points]
    return data, decode_shortcuts

def edge_module(shortcuts, m_fea=256, out_fea=2):
    edge_fea_list = []
    edge_list = []
    out_shape = shortcuts[0].shape[2:]
    shape = fluid.layers.shape(shortcuts[0])[2:]
    
    for i in range(3):
        edge_name='edge'+str(i)+'_'
        data1 = conv_bn_relu(shortcuts[i], m_fea, 
                name=edge_name+'conv1')
        data2 = conv_layer(data1, out_fea, 3, padding=1, 
                name=edge_name+'conv2')
        if i > 0:
            data1 = fluid.layers.resize_bilinear(data1, 
                    out_shape, actual_shape=shape,
                    name=edge_name+'interp1')
            data2 = fluid.layers.resize_bilinear(data2, 
                    out_shape, actual_shape=shape,
                    name=edge_name+'interp2')    
            shape.stop_gradient = True
        edge_fea_list.append(data1)
        edge_list.append(data2)

    edge_cat = fluid.layers.concat(edge_list, axis=1, name='edge_cat')
    edge = conv_layer(edge_cat, out_fea, 1, name='edge_conv')
    edge_fea = fluid.layers.concat(edge_fea_list, axis=1, name='edge_fea_cat')
    return edge, edge_fea

def decoder(psp, shortcut):
    shape = fluid.layers.shape(shortcut)[2:] 
    conv1 = conv_bn_relu(psp, 256, name='decoder_conv1')
    interp1 = fluid.layers.resize_bilinear(conv1, 
            shortcut.shape[2:], actual_shape=shape, name='decoder_interp1')
    shape.stop_gradient = True

    conv2 = conv_bn_relu(shortcut, 48, name='decoder_conv2')
    cat = fluid.layers.concat([interp1, conv2], axis=1, name='decoder_cat')

    conv3 = conv_bn_relu(cat, 256, name='decoder_conv3')
    conv4 = conv_bn_relu(conv3, 256, name='decoder_conv4')

    return conv4
    

def get_logit(input, num_classes, name='logit'):
    # 根据类别数决定最后一层卷积输出
    param_attr = fluid.ParamAttr(
        name=name + '_weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
    out = conv_layer(input, num_classes, param_attr=param_attr, name=name)
    return out


def ACE2P(input, num_classes):
    # Backbone: ResNet
    res, shortcuts = resnet(input)
    psp = psp_module(res, 512)
    edge, edge_fea = edge_module(shortcuts)
    decoder_data = decoder(psp, shortcuts[0])
    seg1 = get_logit(decoder_data, num_classes, name='decoder_logit')

    cat = fluid.layers.concat([decoder_data, edge_fea], axis=1, name='cat')
    conv = conv_bn_relu(cat, 256, name='conv')
    seg2 = get_logit(conv, num_classes)
    return [[seg1, seg2], [edge]]
