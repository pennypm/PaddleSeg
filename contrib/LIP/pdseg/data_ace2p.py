# -*- coding: utf-8 -*- 
import numpy as np
import paddle.fluid as fluid
from utils.config import cfg
import cv2

def get_affine_points(src_shape, dst_shape, rot_grad=0, mirror=0):
    # 获取图像和仿射后图像的三组对应点坐标
    # 三组点为仿射变换后图像的中心点, [w/2,0], [0,0]，及对应原始图像的点
    if dst_shape[0] == 0 or dst_shape[1] == 0:
        raise Exception('scale shape should not be 0')

    # 旋转角度
    rotation = rot_grad * np.pi / 180.0
    sin_v = np.sin(rotation)
    cos_v = np.cos(rotation)

    dst_ratio = float(dst_shape[0]) / dst_shape[1]
    h, w = src_shape
    src_ratio = float(h) / w if w != 0 else 0
    affine_shape = [h, h * dst_ratio] if src_ratio > dst_ratio \
                    else [w / dst_ratio, w]

    # 原始图像三组点
    points = [[0, 0]] * 3
    center = (np.array([w, h]) - 1) * 0.5
    if mirror:
        points[0] = w - center[0] - 1
    else:
        points[0] = center
    points[1] = points[0] + 0.5 * affine_shape[0] * np.array([sin_v, -cos_v])
    points[2] = points[1] - 0.5 * affine_shape[1] * np.array([cos_v, sin_v])

    # 仿射变换后图三组点
    points_trans = [[0, 0]] * 3
    points_trans[0] = (np.array(dst_shape[::-1]) - 1) * 0.5
    points_trans[1] = [points_trans[0][0], 0]

    return points, points_trans

def preprocess(im):
    # ACE2P模型数据预处理
    im_shape = im.shape[:2]
    input_images = []
    for i, scale in enumerate(cfg.TEST.MULTI_SCALES):
        # 获取图像和仿射变换后图像的对应点坐标
        points, points_trans = get_affine_points(im_shape, scale)
        # 根据对应点集获得仿射矩阵
        trans = cv2.getAffineTransform(np.float32(points),
                                       np.float32(points_trans))
        # 根据仿射矩阵对图像进行仿射
        input = cv2.warpAffine(im,
                               trans,
                               scale[::-1],
                               flags=cv2.INTER_LINEAR)

        # 减均值测，除以方差，转换数据格式为NCHW
        input = input.astype(np.float32)
        input = (input / 255. - np.array(cfg.MEAN)) / np.array(cfg.STD)
        input = input.transpose(2, 0, 1).astype(np.float32)
        input = np.expand_dims(input, 0)

        # 水平翻转
        if cfg.AUG.MIRROR:
            flip_input = input[:, :, :, ::-1]
            input_images.append(np.vstack((input, flip_input)))
        else:
            input_images.append(input)

    return input_images


def multi_scale_test(exe, test_prog, feed_name, fetch_list,
                        input_ims, im_shape):
    
    # 由于部分类别分左右部位, flipped_idx为其水平翻转后对应的标签
    flipped_idx = (15, 14, 17, 16, 19, 18)
    ms_outputs = []
    
    # 多尺度预测
    for idx, scale in enumerate(cfg.TEST.MULTI_SCALES):
        input_im = input_ims[idx]
        parsing_output = exe.run(program=test_prog,
                                 feed={feed_name[0]: input_im},
                                 fetch_list=fetch_list)
        output = parsing_output[0][0]
        if cfg.AUG.MIRROR:
            # 若水平翻转，对部分类别进行翻转，与原始预测结果取均值
            flipped_output = parsing_output[0][1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            flipped_output = flipped_output[:, :, ::-1]
            output += flipped_output
            output *= 0.5

        output = np.transpose(output, [1, 2, 0])
        # 仿射变换回图像原始尺寸
        points, points_trans = get_affine_points(im_shape, scale)
        M = cv2.getAffineTransform(np.float32(points_trans), np.float32(points))
        logits_result = cv2.warpAffine(output, M, im_shape[::-1], flags=cv2.INTER_LINEAR)
        ms_outputs.append(logits_result)

    # 多尺度预测结果求均值，求预测概率最大的类别
    ms_fused_parsing_output = np.stack(ms_outputs)
    ms_fused_parsing_output = np.mean(ms_fused_parsing_output, axis=0)
    parsing = np.argmax(ms_fused_parsing_output, axis=2)
    return parsing, ms_fused_parsing_output


def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge


scale_factor=0.25
rotation_factor=30
flip_prob = 0.5
crop_size=[473, 473]
aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
pixel_std=200

def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)

def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)

    return center, scale

def data_aug(im, parsing_anno, person_center, s):
    sf = scale_factor
    rf = rotation_factor
    s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
    r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
        if np.random.random() <= 0.6 else 0

    if np.random.random() <= flip_prob:
        im = im[:, ::-1, :]
        parsing_anno = parsing_anno[:, ::-1]
        person_center[0] = im.shape[1] - person_center[0] - 1
        right_idx = [15, 17, 19]
        left_idx = [14, 16, 18]
        for i in range(0, 3):
            right_pos = np.where(parsing_anno == right_idx[i])
            left_pos = np.where(parsing_anno == left_idx[i])
            parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
            parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]
    return im, parsing_anno, person_center, s, r


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

