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

from __future__ import print_function
import sys
import os
import math
import random
import functools
import io
import time
import codecs

import numpy as np
import paddle
import paddle.fluid as fluid
import cv2

import data_aug as aug
from utils.config import cfg
from data_utils import GeneratorEnqueuer
from models.model_builder import ModelPhase
from data_ace2p import get_affine_points, _box2cs, get_affine_transform, data_aug, generate_edge

def cv2_imread(file_path, flag=cv2.IMREAD_COLOR):
    # resolve cv2.imread open Chinese file path issues on Windows Platform.
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


class SegDataset(object):
    def __init__(self,
                 file_list,
                 data_dir,
                 shuffle=False,
                 mode=ModelPhase.TRAIN):
        self.mode = mode
        self.shuffle = shuffle
        self.data_dir = data_dir

        # NOTE: Please ensure file list was save in UTF-8 coding format
        with codecs.open(file_list, 'r', 'utf-8') as flist:
            self.lines = [line.strip() for line in flist]
            if shuffle:
                np.random.shuffle(self.lines)

    def generator(self):
        if self.shuffle:
            np.random.shuffle(self.lines)
        for line in self.lines:
            yield self.process_image(line, self.data_dir, self.mode)

    def sharding_generator(self, pid=0, num_processes=1):
        """
        Use line id as shard key for multiprocess io
        It's a normal generator if pid=0, num_processes=1
        """
        for index, line in enumerate(self.lines):
            # Use index and pid to shard file list
            if index % num_processes == pid:
                yield self.process_image(line, self.data_dir, self.mode)

    def batch_reader(self, batch_size):
        br = self.batch(self.reader, batch_size)
        for batch in br:
            yield batch[0], batch[1], batch[2]

    def multiprocess_generator(self, max_queue_size=32, num_processes=8):
        # Re-shuffle file list
        if self.shuffle:
            np.random.shuffle(self.lines)
        # Create multiple sharding generators according to num_processes for multiple processes
        generators = []
        for pid in range(num_processes):
            generators.append(self.sharding_generator(pid, num_processes))

        try:
            enqueuer = GeneratorEnqueuer(generators)
            enqueuer.start(max_queue_size=max_queue_size, workers=num_processes)
            while True:
                generator_out = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_out = enqueuer.queue.get(timeout=5)
                        break
                    else:
                        time.sleep(0.01)
                if generator_out is None:
                    break
                yield generator_out
        finally:
            if enqueuer is not None:
                enqueuer.stop()

    def batch(self, reader, batch_size, is_test=False, drop_last=False):
        def batch_reader(is_test=False, drop_last=drop_last):
            if is_test:
                imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []
                for img, grt, img_name, valid_shape, org_shape in reader():
                    imgs.append(img)
                    grts.append(grt)
                    img_names.append(img_name)
                    valid_shapes.append(valid_shape)
                    org_shapes.append(org_shape)
                    if len(imgs) == batch_size:
                        yield np.array(imgs), np.array(
                            grts), img_names, np.array(valid_shapes), np.array(
                                org_shapes)
                        imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []

                if not drop_last and len(imgs) > 0:
                    yield np.array(imgs), np.array(grts), img_names, np.array(
                        valid_shapes), np.array(org_shapes)
            else:
                imgs, labs, lab_edges, ignore = [], [], []
                bs = 0
                for img, lab, lab_edge, ig in reader():
                    imgs.append(img)
                    labs.append(lab)
                    lab_edges.append(lab_edge)
                    ignore.append(ig)
                    bs += 1
                    if bs == batch_size:
                        yield np.array(imgs), np.array(labs), np.array(lab_edges), np.array(ignore)
                        bs = 0
                        imgs, labs, lab_edges, ignore = [], [], []

                if not drop_last and bs > 0:
                    yield np.array(imgs), np.array(labs), np.array(lab_edges), np.array(ignore)

        return batch_reader(is_test, drop_last)

    def load_image(self, line, src_dir, mode=ModelPhase.TRAIN):
        # original image cv2.imread flag setting
        cv2_imread_flag = cv2.IMREAD_COLOR
        if cfg.DATASET.IMAGE_TYPE == "rgba":
            # If use RBGA 4 channel ImageType, use IMREAD_UNCHANGED flags to
            # reserver alpha channel
            cv2_imread_flag = cv2.IMREAD_UNCHANGED

        parts = line.strip().split(cfg.DATASET.SEPARATOR)
        if len(parts) != 2:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception("File list format incorrect! It should be"
                                " image_name{}label_name\\n".format(
                                    cfg.DATASET.SEPARATOR))
            img_name, grt_name = parts[0], None
        else:
            img_name, grt_name = parts[0], parts[1]

        img_path = os.path.join(src_dir, img_name)
        img = cv2_imread(img_path, cv2_imread_flag)

        if grt_name is not None:
            grt_path = os.path.join(src_dir, grt_name)
            grt = cv2_imread(grt_path, cv2.IMREAD_GRAYSCALE)
        else:
            grt = None

        if img is None:
            raise Exception(
                "Empty image, src_dir: {}, img: {} & lab: {}".format(
                    src_dir, img_path, grt_path))

        img_height = img.shape[0]
        img_width = img.shape[1]

        if grt is not None:
            grt_height = grt.shape[0]
            grt_width = grt.shape[1]

            if img_height != grt_height or img_width != grt_width:
                raise Exception(
                    "source img and label img must has the same size")
        else:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception(
                    "Empty image, src_dir: {}, img: {} & lab: {}".format(
                        src_dir, img_path, grt_path))

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_channels = img.shape[2]
        if img_channels < 3:
            raise Exception("PaddleSeg only supports gray, rgb or rgba image")
        if img_channels != cfg.DATASET.DATA_DIM:
            raise Exception(
                "Input image channel({}) is not match cfg.DATASET.DATA_DIM({}), img_name={}"
                .format(img_channels, cfg.DATASET.DATADIM, img_name))
        if img_channels != len(cfg.MEAN):
            raise Exception(
                "img name {}, img chns {} mean size {}, size unequal".format(
                    img_name, img_channels, len(cfg.MEAN)))
        if img_channels != len(cfg.STD):
            raise Exception(
                "img name {}, img chns {} std size {}, size unequal".format(
                    img_name, img_channels, len(cfg.STD)))

        return img, grt, img_name, grt_name

    def normalize_image(self, img):
        """ 像素归一化后减均值除方差 """
        img = img.transpose((2, 0, 1)).astype('float32') / 255.0
        img_mean = np.array(cfg.MEAN).reshape((len(cfg.MEAN), 1, 1))
        img_std = np.array(cfg.STD).reshape((len(cfg.STD), 1, 1))
        img -= img_mean
        img /= img_std

        return img

    def process_image(self, line, data_dir, mode):
        """ process_image """
        img, grt, img_name, grt_name = self.load_image(
            line, data_dir, mode=mode)
        im_shape = img.shape[:2]

        crop_size = cfg.TRAIN_CROP_SIZE #[473, 473]
        h, w = img.shape[:2]
        person_center, s = _box2cs([0, 0, w - 1, h - 1])
        r = 0
        if mode == ModelPhase.TRAIN:
            img, grt, person_center, s, r = data_aug(img, grt, person_center, s)

        trans = get_affine_transform(person_center, s, r, crop_size)
        img = cv2.warpAffine(img, trans, (int(crop_size[1]), int(crop_size[0])),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        #points, points_trans = get_affine_points(im_shape, crop_size, r, mirror)
        #trans = cv2.getAffineTransform(np.float32(points), np.float32(points_trans))
        #img = cv2.warpAffine(img, trans, tuple(crop_size[::-1]), flags=cv2.INTER_LINEAR)
    
        # Normalize image
        img = self.normalize_image(img)

        if ModelPhase.is_train(mode) or ModelPhase.is_eval(mode):
            grt = cv2.warpAffine(grt, trans, tuple(crop_size[::-1]),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(255))
            label_edge = generate_edge(grt)
            grt = np.expand_dims(np.array(grt).astype('int32'), axis=0)
            label_edge = np.expand_dims(np.array(label_edge).astype('int32'), axis=0)
            ignore = (grt != cfg.DATASET.IGNORE_INDEX).astype('int32')

        if ModelPhase.is_train(mode):
            return (img, grt, label_edge, ignore)
        elif ModelPhase.is_eval(mode):
            return (img, grt, ignore)
        elif ModelPhase.is_visual(mode):
            valid_shape = cfg.EVAL_CROP_SIZE
            org_shape = im_shape
            return (img, grt, img_name, valid_shape, org_shape)
    
