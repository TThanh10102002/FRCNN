import numpy as np
import tensorflow as tf


#Compute IoU for multibox parallel
def iou(box1, box2):
    """
      Computes intersection-over-union (IoU) for multiple boxes in parallel.
      Parameters
      ----------
      boxes1 : np.ndarray
        Box corners, shaped (N, 4), with each box as (y1, x1, y2, x2).
      boxes2 : np.ndarray
        Box corners, shaped (M, 4).
      Returns
      -------
      np.ndarray
        IoUs for each pair of boxes in boxes1 and boxes2, shaped (N, M).
    """
    # (N,1,2) and (M,2) -> (N,M,2) indicating top-left corners of box pairs
    top_left_point = np.maximum(box1[: , None , 0:2], box2[: , 0:2])
    bottom_right_point = np.minimum(box1[: , None , 2:4], box2[: , 2:4])
    
    # (N,M) indicating whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
    ord_mask = np.all(top_left_point < bottom_right_point, axis = 2)
    
    # (N,M) indicating intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    intersection = ord_mask * np.prod(bottom_right_point - top_left_point, axis = 2)
    
    # (N,) indicating areas of boxes1
    area_1 = np.prod(box1[:, 2:4] - box1[:, 0:2], axis = 1)
    
    # (M,) indicating areas of boxes2
    area_2 = np.prod(box2[:, 2:4] - box2[:, 0:2], axis = 1)
    
     # (N,1) + (M,) - (N,M) = (N,M), union areas of both boxes
    union_area = area_1[:, None] + area_2 - intersection
    eps = 1e-7
    return intersection / (union_area + eps)


def tf_iou(box1, box2):
    """
    Equivalent of intersection_over_union() but operates on tf.Tensors and
    produces a TensorFlow graph suitable for use in a model. This code borrowed
    from Matterport's MaskRCNN implementation:
    https://github.com/matterport/Mask_RCNN
    Parameters
    ----------
    boxes1: tf.Tensor
        Box corners, shaped (N,4), with each box as (y1, x1, y2, x2).
    boxes2: tf.Tensor
        Box corners, shaped (M,4).
    Returns
    -------
    tf.Tensor
        Tensor of shape (N, M) containing IoU score between each pair of boxes.
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    box_1 = tf.reshape(tf.tile(tf.expand_dims(box1, 1), [1, 1, tf.shape(box2)[0]]), [-1, 4])
    box_2 = tf.tile(box2, [tf.shape(box1)[0], 1])
    #Compute intersection
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(box_1, 4, axis = 1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(box_2, 4, axis = 1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    #Compute unions
    box1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    box2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = box1_area + box2_area - intersection
    #Compute IoU and reshape to [box1, box2]
    iou_point = intersection / union
    overlap_area = tf.reshape(iou_point, [tf.shape(box1)[0], tf.shape(box2)[0]])
    return overlap_area


def convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
    """
    Converts box deltas, which are in parameterized form (ty, tx, th, tw) as
    described by the Fast R-CNN and Faster R-CNN papers, to boxes
    (y1, x1, y2, x2). The anchors are the base boxes (e.g., RPN anchors or
    proposals) that the deltas describe a modification to.
    Parameters
    ----------
    box_deltas : np.ndarray
        Box deltas with shape (N, 4). Each row is (ty, tx, th, tw).
    anchors : np.ndarray
        Corresponding anchors that the deltas are based upon, shaped (N, 4) with
        each row being (center_y, center_x, height, width).
    box_delta_means : np.ndarray
        Mean ajustment to deltas, (4,), to be added after standard deviation
        scaling and before conversion to actual box coordinates.
    box_delta_stds : np.ndarray
        Standard deviation adjustment to deltas, (4,). Box deltas are first
        multiplied by these values.
    Returns
    -------
    np.ndarray
        Box coordinates, (N, 4), with each row being (y1, x1, y2, x2).
    """
    box_deltas = box_deltas * box_delta_stds + box_delta_means
    # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
    center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2]  
    # width = anchor_width * exp(tw), height = anchor_height * exp(th)
    size = anchors[:,2:4] * np.exp(box_deltas[:,2:4])             
    boxes = np.empty(box_deltas.shape)
    # y1, x1
    boxes[:,0:2] = center - 0.5 * size  
    # y2, x2                          
    boxes[:,2:4] = center + 0.5 * size                            
    return boxes


def tf_convert_deltas_to_boxes(box_deltas, anchors, box_delta_means, box_delta_stds):
    """
    Equivalent of convert_deltas_to_boxes() but operates on tf.Tensors and
    produces a TensorFlow graph suitable for use in a model.
    Parameters
    ----------
    box_deltas : np.ndarray
        Box deltas with shape (N, 4). Each row is (ty, tx, th, tw).
    anchors : np.ndarray
        Corresponding anchors that the deltas are based upon, shaped (N, 4) with
        each row being (center_y, center_x, height, width).
    box_delta_means : np.ndarray
        Mean ajustment to deltas, (4,), to be added after standard deviation
        scaling and before conversion to actual box coordinates.
    box_delta_stds : np.ndarray
        Standard deviation adjustment to deltas, (4,). Box deltas are first
        multiplied by these values.
    Returns
    -------
    tf.Tensor
        Box coordinates, (N, 4), with each row being (y1, x1, y2, x2).
    """
    box_deltas = box_deltas * box_delta_stds + box_delta_means
    # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
    center = anchors[:,2:4] * box_deltas[:,0:2] + anchors[:,0:2] 
    # width = anchor_width * exp(tw), height = anchor_height * exp(th) 
    size = anchors[:,2:4] * tf.math.exp(box_deltas[:,2:4]) 
    # y1, x1       
    boxes_top_left = center - 0.5 * size 
    # y2, x2                        
    boxes_bottom_right = center + 0.5 * size     
    # [ (N,2), (N,2) ] -> (N,4)                 
    boxes = tf.concat([boxes_top_left, boxes_bottom_right], axis = 1) 
    return boxes
