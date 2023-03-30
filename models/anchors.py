# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/anchors.py
# Copyright 2021-2022 Bart Trzynadlowski

import itertools
from math import sqrt
import numpy as np
from . import math


def _compute_anchor_size():
    #
    # Anchor scales and aspect ratios.
    #
    # x * y = area          x * (x_aspect * x) = x_aspect * x^2 = area
    # x_aspect * x = y  ->  x = sqrt(area / x_aspect)
    #                       y = x_aspect * sqrt(area / x_aspect)
    #
    areas = [128*128, 256*256, 512*512]
    x_aspects = [0.5, 1.0, 2.0]

    # Generate all 9 combinations of area and aspect ratio
    heights = np.array([x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])
    width = np.array([sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3))])

    # Return as (9,2) matrix of sizes
    return np.vstack([heights, width]).T


def generate_anchor_maps(img_shape, f_pixels):
    """
    Generates maps defining the anchors for a given input image size. There are 9
    different anchors at each feature map cell (3 scales, 3 ratios).
    Parameters
    ----------
    image_shape : Tuple[int, int, int]
        Shape of the input image, (height, width, channels), at the scale it will
        be passed into the Faster R-CNN model.
    feature_pixels : int
        Distance in pixels between anchors. This is the size, in input image space,
        of each cell of the feature map output by the feature extractor stage of
        the Faster R-CNN network.
    Returns
    -------
    np.ndarray, np.ndarray
        Two maps, with height and width corresponding to the feature map
        dimensions, not the input image:
        1. A map of shape (height, width, num_anchors*4) containing all anchors,
            each stored as (center_y, center_x, anchor_height, anchor_width) in
            input image pixel space.
        2. A map of shape (height, width, num_anchors) indicating which anchors
            are valid (1) or invalid (0). Invalid anchors are those that cross
            image boundaries and must not be used during training.
    """
    assert len(img_shape) == 3

    # Base anchor template: (num_anchors,4), with each anchor being specified by
    # its corners (y1,x1,y2,x2)
    anchors_size = _compute_anchor_size()
    anchors_num = anchors_size.shape[0]
    anchors_template = np.empty((anchors_num, 4))
    anchors_template[:, 0:2] = -0.5 * anchors_size
    anchors_template[:, 2:4] = 0.5 * anchors_size

    # Shape of map, (H,W), determined by VGG-16 backbone
    height, width = img_shape[0] // f_pixels, img_shape[1] // f_pixels

     # Generate (H,W,2) map of coordinates, in feature space, each being [y,x]
    x_coord = np.arange(width)
    y_coord = np.arange(height)
    cell_coord = np.array(np.meshgrid(y_coord, x_coord)).transpose([2, 1, 0])

    # Convert all coordinates to image space (pixels) at *center* of each cell
    center = cell_coord * f_pixels + 0.5 * f_pixels

    # (H,W,2) -> (H,W,4), repeating the last dimension so it contains (y,x,y,x)
    center = np.tile(center, reps=2)

    # (H,W,2) -> (H,W,4*anchors_num)
    center = np.tile(center, reps=anchors_num)

    anchors = center.astype(np.float32) + anchors_template.flatten()

    # (H,W,4*num_anchors) -> (H*W*num_anchors,4)
    anchors = anchors.reshape((height * width * anchors_num, 4))

    # Valid anchors are those that do not cross image boundaries
    img_height, img_width = img_shape[0 : 2]

    valid_anchors = np.all((anchors[:, 0:2] >= [0, 0]) & (anchors[:, 2:4] <= [img_height, img_width]), axis = 1)

    # Convert anchors to anchor format: (center_y, center_x, height, width)
    anchor_map = np.empty((anchors.shape[0], 4))
    anchor_map[:, 0:2] = 0.5 * (anchors[:, 0:2] + anchors[:, 2:4])
    anchor_map[:, 2:4] = anchors[:, 2:4] - anchors[:, 0:2]

    # Reshape maps and return
    anchor_map = anchor_map.reshape((height, width, 4 * anchors_num))
    anchors_valid_map = valid_anchors.reshape((height, width, anchors_num))
    return anchor_map.astype(np.float32), anchors_valid_map.astype(np.float32)


def generate_rpn_map(anchor_map, anchor_valid_map, gt_boxes, obj_iou_threshold = 0.7, background_iou_threshold = 0.3):
    """
    Generates a map containing ground truth data for training the region proposal
    network.
    Parameters
    ----------
    anchor_map : np.ndarray
        Map of shape (height, width, num_anchors*4) defining the anchors as
        (center_y, center_x, anchor_height, anchor_width) in input image space.
    anchor_valid_map : np.ndarray
        Map of shape (height, width, num_anchors) defining anchors that are valid
        and may be included in training.
    gt_boxes : List[training_sample.Box]
        List of ground truth boxes.
    object_iou_threshold : float
        IoU threshold between an anchor and a ground truth box above which an
        anchor is labeled as an object (positive) anchor.
    background_iou_threshold : float
        IoU threshold below which an anchor is labeled as background (negative).
    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        RPN ground truth map, object (positive) anchor indices, and background
        (negative) anchor indices. Map height and width dimensions are in feature
        space.
        1. RPN ground truth map of shape (height, width, num_anchors, 6) where the
        last dimension is:
        - 0: Trainable anchor (1) or not (0). Only valid and non-neutral (that
                is, definitely positive or negative) anchors are trainable. This is
                the same as anchor_valid_map with additional invalid anchors caused
                by neutral samples
        - 1: For trainable anchors, whether the anchor is an object anchor (1)
                or background anchor (0). For non-trainable anchors, will be 0.
        - 2: Regression target for box center, ty.
        - 3: Regression target for box center, tx.
        - 4: Regression target for box size, th.
        - 5: Regression target for box size, tw.
        2. Map of shape (N, 3) of indices (y, x, k) of all N object anchors in the
        RPN ground truth map.
        3. Map of shape (M, 3) of indices of all M background anchors in the RPN
        ground truth map.
    """
    height, width, anchor_num = anchor_valid_map.shape

    # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
    gt_box_corners = np.array([box.corners for box in gt_boxes])
    gt_box_num = len(gt_boxes)

    # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
    gt_box_centers = 0.5 * (gt_box_corners[:, 0:2] + gt_box_corners[:, 2:4])
    gt_box_sides = gt_box_corners[:, 2:4] - gt_box_corners[:, 0:2]

    # Flatten anchor boxes to (N,4) and convert to corners
    anchor_map = anchor_map.reshape((-1, 4))
    anchors = np.empty(anchor_map.shape)
    anchors[:, 0:2] = anchor_map[:, 0:2] - 0.5 * anchor_map[:, 2:4]
    anchors[:, 2:4] = anchor_map[:, 0:2] + 0.5 * anchor_map[:, 2:4]
    box_num = anchors.shape[0]

    # RPN class: 0 = background, 1 = foreground, -1 = ignore (these will be marked as invalid in the truth map)
    obj_score = np.full(box_num, -1)

    # -1 means no box
    gt_box_assign = np.full(box_num, -1)

    # Compute IoU between each anchor and each ground truth box, (N,M).
    ious = math.iou(box1=anchors, box2=gt_box_corners)

    #wipe IoU scores of invalid anchors (straddle image boundaries)
    ious[anchor_valid_map.flatten() == 0, :] = -1.0

    # Find the best IoU ground truth box for each anchor and the best IoU anchor
    # for each ground truth box.
    #
    # Note that ious == max_iou_per_gt_box tests each of the N rows of ious
    # against the M elements of max_iou_per_gt_box, column-wise. np.where() then
    # returns all (y,x) indices of matches as a tuple: (y_indices, x_indices).
    # The y indices correspond to the N dimension and therefore indicate anchors
    # and the x indices correspond to the M dimension (ground truth boxes).
    max_iou_per_anchor = np.max(ious, axis = 1)
    best_box_idx_per_anchor = np.argmax(ious, axis = 1)
    max_iou_per_gt_box = np.max(ious, axis = 0)
    highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0]

    # Anchors below the minimum threshold are negative
    obj_score[max_iou_per_anchor < background_iou_threshold] = 0

    # Anchors that meet the threshold IoU are positive
    obj_score[max_iou_per_anchor >= obj_iou_threshold] = 1

    # Anchors that overlap the most with ground truth boxes are positive
    obj_score[highest_iou_anchor_idxs] = 1

    # We assign the highest IoU ground truth box to each anchor. If no box met
    # the IoU threshold, the highest IoU box may happen to be a box for which
    # the anchor had the highest IoU. If not, then the objectness score will be
    # negative and the box regression won't ever be used.
    gt_box_assign[:] = best_box_idx_per_anchor

    # Anchors that are to be ignored will be marked invalid. Generate a mask to
    # multiply anchor_valid_map by (-1 -> 0, 0 or 1 -> 1). Then mark ignored
    # anchors as 0 in objectness score because the score can only really be 0 or
    # 1.
    mask = (obj_score >= 0).astype(np.float32)
    obj_score[obj_score < 0] = 0

    # Compute box delta regression targets for each anchor
    box_delta_targets = np.empty((box_num, 4))
    # ty = (box_center_y - anchor_center_y) / anchor_height, tx = (box_center_x - anchor_center_x) / anchor_width
    box_delta_targets[:, 0:2] = (gt_box_centers[gt_box_assign] - anchor_map[:, 0:2]) / anchor_map[:, 2:4]
    # th = log(box_height / anchor_height), tw = log(box_width / anchor_width)
    box_delta_targets[:, 2:4] = np.log(gt_box_sides[gt_box_assign] / anchor_map[:, 2:4])

    # Assemble RPN ground truth map
    rpn_map = np.zeros((height, width, anchor_num, 6))
    # trainable anchors (object or background; excludes boundary-crossing invalid and neutral anchors)
    rpn_map[:, :, :, 0] = anchor_valid_map * mask.reshape((height, width, anchor_num))
    rpn_map[:, :, :, 1] = obj_score.reshape((height, width, anchor_num))
    rpn_map[:, :, :, 2:6] = box_delta_targets.reshape((height, width, anchor_num, 4))

    # Return map along with positive and negative anchors
    # shape (height,width,k,3): every index (y,x,k,:) returns its own coordinate (y,x,k)
    rpn_map_coord = np.transpose(np.mgrid[0:height, 0:width, 0:anchor_num], (1, 2, 3, 0))
    # shape (N,3), where each row is the coordinate (y,x,k) of a positive sample
    obj_anchor_idxs = rpn_map_coord[np.where((rpn_map[:, :, :, 1] > 0) & (rpn_map[:, :, :, 0] > 0))]
    # shape (N,3), where each row is the coordinate (y,x,k) of a negative sample
    background_anchor_idxs = rpn_map_coord[np.where((rpn_map[:, :, :, 1] == 0) & (rpn_map[:, :, :, 0] > 0))]

    return rpn_map.astype(np.float32), obj_anchor_idxs, background_anchor_idxs




