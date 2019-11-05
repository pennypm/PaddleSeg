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

import sys
import paddle.fluid as fluid
import numpy as np
import importlib
from utils.config import cfg

def weigthed_softmax_with_cross_entropy(input, label, weights, class_num, 
                                    ignore_index, ignore_mask):
    input = fluid.layers.softmax(input, axis=1)
    log_input = fluid.layers.log(input)
    label_onehot = fluid.layers.one_hot(label, class_num)
    loss = fluid.layers.matmul(-1.0 * label_onehot * log_input, weights)
    loss *= (label!=ignore_index)
    base = fluid.layers.matmul(label_onehot, weights)
    base *= (label!=ignore_index)
    loss = loss * ignore_mask
    base = base * ignore_mask
    avg_loss = fluid.layers.reduce_sum(loss) / fluid.layers.reduce_sum(base)    
    return avg_loss

def softmax_with_loss(logit, label, ignore_mask=None, num_classes=2, weights=None):

    ignore_mask = fluid.layers.cast(ignore_mask, 'float32')
    label = fluid.layers.elementwise_min(
        label, fluid.layers.assign(np.array([num_classes - 1], dtype=np.int32)))
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    ignore_mask = fluid.layers.reshape(ignore_mask, [-1, 1])
    
    if weights:
        avg_loss = weigthed_softmax_with_cross_entropy(logit, label, weights, 
                    num_classes, cfg.DATASET.IGNORE_INDEX, ignore_mask)
    else:
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=cfg.DATASET.IGNORE_INDEX,
            return_softmax=True)

        loss = loss * ignore_mask
        if cfg.MODEL.FP16:
            loss = fluid.layers.cast(loss, 'float32')
            avg_loss = fluid.layers.mean(loss) / fluid.layers.mean(ignore_mask)
            avg_loss = fluid.layers.cast(avg_loss, 'float16')
        else:
           avg_loss = fluid.layers.mean(loss) / fluid.layers.mean(ignore_mask)
        if cfg.MODEL.SCALE_LOSS > 1.0:
            avg_loss = avg_loss * cfg.MODEL.SCALE_LOSS
    
    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return avg_loss

def mulogit_onelabel_loss(loss, logits, label, ignore_mask=None, num_classes=2, weights=None):
    if isinstance(logits, list):
        for i, logit in enumerate(logits):
            logit = fluid.layers.resize_bilinear(logit, label.shape[2:])
            logit_mask = (label.astype('int32') !=
                          cfg.DATASET.IGNORE_INDEX).astype('int32')
            loss += softmax_with_loss(logit, label, logit_mask, num_classes, weights)
    else:
        logits = fluid.layers.resize_bilinear(logits, label.shape[2:])
        loss += softmax_with_loss(logits, label, ignore_mask, num_classes, weights)
    return loss

def multi_softmax_with_loss(logits, label, ignore_mask=None, num_classes=2):
    loss = 0
    logit_seg, logit_edge = logits
    label_seg, label_edge = label
    loss = mulogit_onelabel_loss(loss, logit_seg, label_seg, ignore_mask, num_classes)

    pos_num = fluid.layers.reduce_sum(fluid.layers.cast(label_edge==1, 'float32'))
    neg_num = fluid.layers.reduce_sum(fluid.layers.cast(label_edge==0, 'float32'))
    weight_pos = neg_num / (pos_num + neg_num)
    weights = fluid.layers.concat([1 - weight_pos, weight_pos])
    weights = fluid.layers.unsqueeze(weights, axes=[1])

    loss = mulogit_onelabel_loss(loss, logit_edge, label_edge, ignore_mask, num_classes=2, weights=weights) 
    return loss


# to change, how to appicate ignore index and ignore mask
def dice_loss(logit, label, ignore_mask=None, num_classes=2):
    if num_classes != 2:
        raise Exception("dice loss is only applicable to binary classfication")
    ignore_mask = fluid.layers.cast(ignore_mask, 'float32')
    label = fluid.layers.elementwise_min(
        label, fluid.layers.assign(np.array([num_classes - 1], dtype=np.int32)))
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    logit = fluid.layers.softmax(logit)
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    ignore_mask = fluid.layers.reshape(ignore_mask, [-1, 1])
    loss = fluid.layers.dice_loss(logit, label)
    return loss
