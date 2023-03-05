import itertools
from math import sqrt
import numpy as np
from . import math


def _compute_anchor_size():
    areas = [128*128, 256*256, 512*512]
    x_aspects = [0.5, 1.0, 2.0]

    heights = np.array([x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])
    width = np.array([sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])

    return np.vstack([heights, width]).T


def generate_anchor_maps(img_shape, f_pixels):
    assert img_shape == 3
    anchors_size = _compute_anchor_size()
    anchors_num = anchors_size.shape[0]
    anchors_template = np.empty((anchors_num, 4))
    anchors_template[:, 0:2] = -0.5 * anchors_size
    anchors_template[:, 2:4] = 0.5 * anchors_size

    height, width = img_shape[0] // f_pixels, img_shape[1] // f_pixels

    x_coord = np.arange(width)
    y_coord = np.arange(height)
    cell_coord = np.arange(np.meshgrid(y_coord, x_coord)).transpose([2, 1, 0])

    center = cell_coord * f_pixels + 0.5 * f_pixels

    center = np.tile(center, reps=2)

    center = np.tile(center, reps=anchors_num)

    anchors = center.astype(np.float32) + anchors_template.flatten()

    anchors = anchors.reshape((height * width * anchors_num, 4))

    img_height, img_width = img_shape[0 : 2]

    valid_anchors = np.all((anchors[:, 0:2] >= [0, 0]) & (anchors[:, 2:4] <= [img_height, img_width]), axis = 1)

    anchor_map = np.empty((anchors.shape[0], 4))
    anchor_map[:, 0:2] = 0.5 * (anchors[:, 0:2] + anchors[:, 2:4])
    anchor_map[:, 2:4] = anchors[:, 2:4] - anchors[:, 0:2]

    anchor_map = anchor_map.reshape((height, width, 4 * anchors_num))
    anchors_valid_map = valid_anchors.reshape((height, width, anchors_num))
    return anchor_map.astype(np.float32), anchors_valid_map.astype(np.float32)


def generate_rpn_map(anchor_map, anchor_valid_map, gt_boxes, obj_iou_threshold = 0.7, background_iou_threshold = 0.3):
    height, width, anchor_num = anchor_valid_map.shape

    gt_box_corners = np.array([box.corners for box in gt_boxes])
    gt_box_num = len(gt_boxes)

    gt_box_centers = 0.5 * (gt_box_corners[:, 0:2] + gt_box_corners[:, 2:4])
    gt_box_sides = gt_box_corners[:, 2:4] - gt_box_corners[:, 0:2]

    anchor_map = anchor_map.reshape((-1, 4))
    anchors = np.empty(anchor_map.shape)
    anchors[:, 0:2] = anchor_map[:, 0:2] - 0.5 * anchor_map[:, 2:4]
    anchors[:, 2:4] = anchor_map[:, 0:2] + 0.5 * anchor_map[:, 2:4]
    box_num = anchors.shape[0]

    obj_score = np.full(box_num, -1)
    gt_box_assign = np.full(box_num, -1)

    ious = math.iou(box1=anchors, box2=gt_box_corners)

    ious[anchor_valid_map.flatten() == 0, :] = -1.0

    max_iou_per_anchor = np.max(ious, axis = 1)
    best_box_idx_per_anchor = np.argmax(ious, axis = 1)
    max_iou_per_gt_box = np.max(ious, axis = 0)
    highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0]

    obj_score[max_iou_per_anchor < background_iou_threshold] = 0

    obj_score[max_iou_per_anchor >= obj_iou_threshold] = 1

    obj_score[highest_iou_anchor_idxs] = 1

    gt_box_assign[:] = best_box_idx_per_anchor

    mask = (obj_score >= 0).astype(np.float32)
    obj_score[obj_score < 0] = 0

    box_delta_targets = np.empty((n, 4))
    box_delta_targets[:, 0:2] = (gt_box_centers[gt_box_assign] - anchor_map[:, 0:2]) / anchor_map[:, 2:4]
    box_delta_targets[:, 2:4] = np.log(gt_box_sides[gt_box_assign] / anchor_map[:, 2:4])

    rpn_map = np.zeros((height, width, anchor_num, 6))
    rpn_map[:, :, :, 0] = anchor_valid_map * mask.reshape((height, width, anchor_num))
    rpn_map[:, :, :, 1] = obj_score.reshape((height, width, anchor_num))
    rpn_map[:, :, :, 2:6] = box_delta_targets.reshape((height, width, anchor_num, 4))

    rpn_map_coord = np.transpose(np.mgrid[0:height, 0:width, 0:anchor_num], (1, 2, 3, 0))
    obj_anchor_idxs = rpn_map_coord[np.where((rpn_map[:, :, :, 1] > 0) & (rpn_map[:, :, :, 0] > 0))]
    background_anchor_idxs = rpn_map_coord[np.where((rpn_map[:, :, :, 1] == 0) & (rpn_map[:, :, :, 0] > 0))]

    return rpn_map.astype(np.float32), obj_anchor_idxs, background_anchor_idxs




