#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/models/faster_rcnn.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# TensorFlow/Keras implementation of Faster R-CNN training and inference
# models. Here, all stages of Faster R-CNN are instantiated, ground truth
# labels from RPN proposal boxes (RoIs) for the detector stage are generated,
# and proposals are sampled.
#

#
# Weight Decay
# ------------
# Keras does not provide a weight decay option but rather an L2 penalty. Weight
# decay can be converted to L2 by dividing by 2. This is because the L2 penalty
# is added to the loss and then differentiated with respect to the weights
# (introducing a factor of 2 that must be canceled out). See:
# https://bbabenko.github.io/weight-decay/
#
# Pro-Tip
# -------
#
# To log the output of Keras layers using tf.print, use K.Lambda as below:
#
#   def do_log1(x):
#     tf.print("best_ious=", x, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
#     return x
#   best_ious = Lambda(do_log1)(best_ious)
#
#   def do_log(x):
#     y_predicted = x[0]
#     y_true = x[1]
#     loss = K.mean(K.categorical_crossentropy(target = y_true, output = y_predicted, from_logits = True))
#     tf.print("loss=", loss, "y_predicted=", y_predicted, output_stream = "file:///projects/FasterRCNN/tf2/out.txt", summarize = -1)
#     return y_predicted
#   y_predicted = Lambda(do_log)((y_predicted, y_true))
#
# output_stream may also be a file stream like sys.stdout.
#

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Lambda
from keras.applications import VGG16

from . import vgg16
from . import RPN
from . import detector
from . import math

class FasterRCNN(Model):
    def __init__(self, num_classes, allow_edge_proposals, custom_roi_pool, activate_class_outputs, l2 = 0, dropout = 0):
        super().__init__()
        self._num_classes = num_classes
        self._activate_class_outputs = activate_class_outputs
        self._stage1_feature_extractor = vgg16.FeatureExtractor(l2 = l2)
        self._stage2_region_proposal_network = RPN.RegionProposalNetwork(max_proposals_pre_nms_train = 12000,
                                                                         max_proposals_pre_nms_infer = 6000,
                                                                         max_proposals_post_nms_train = 2000,
                                                                         max_proposals_post_nms_infer = 300,
                                                                         l2 = l2, allow_edge_proposals = allow_edge_proposals)
        self._stage3_detector_network = detector.DetectorNet(num_classes = num_classes,
                                                             custom_roi_pool = custom_roi_pool,
                                                             activate_class_outputs = activate_class_outputs,
                                                             l2 = l2, dropout = dropout)
        
    def call(self, inputs, training = False):
        input_img = inputs[0]
        anchor_map = inputs[1]
        anchor_valid_map = inputs[2]
        if training:
            gt_rpn_map = inputs[3]
            gt_box_class_indexes_map = inputs[4]
            gt_box_corners_map = inputs[5]

        #Stage 1: Feature Extraction
        feature_map = self._stage1_feature_extractor(input_img = input_img, training = training)

        #Stage 2: RPN generates proposal map for object
        rpn_scores, rpn_box_deltas, proposals = self._stage2_region_proposal_network(
                                                    inputs = [input_img,
                                                            feature_map,
                                                            anchor_map,
                                                            anchor_valid_map],
                                                    training = training)
        # If training, we must generate ground truth data for the detector stage
        # from RPN outputs
        if training:
            # Assign labels to proposals and take random sample (for detector training)
            proposals, gt_classes, gt_box_deltas = self._label_proposals(proposals = proposals,
                                                                        gt_box_class_indexes = gt_box_class_indexes_map[0],
                                                                        gt_box_corners = gt_box_corners_map[0],
                                                                        min_background_iou_threshold = 0.0,
                                                                        min_object_iou_threshold = 0.5)
            proposals, gt_classes, gt_box_deltas = self._sample_proposals(proposals = proposals,
                                                                          gt_classes = gt_classes,
                                                                          gt_box_deltas = gt_box_deltas,
                                                                          max_proposals = 128,
                                                                          pos_fraction = 0.25)
            gt_classes = tf.expand_dims(gt_classes, axis = 0)
            gt_box_deltas = tf.expand_dims(gt_box_deltas, axis = 0)

            # Ensure proposals are treated as constants and do not propagate gradients
            proposals = tf.stop_gradient(proposals)
            gt_classes = tf.stop_gradient(gt_classes)
            gt_box_deltas = tf.stop_gradient(gt_box_deltas)

        #Stage 3: Detector
        detector_classes, detector_box_deltas = self._stage3_detector_network(inputs = [input_img,
                                                                                        feature_map,
                                                                                        proposals],
                                                                            training = training)

        #Losses
        if training:
            rpn_class_loss = self._stage2_region_proposal_network.class_loss(y_pred = rpn_scores, gt_rpn_map = gt_rpn_map)
            rpn_reg_loss = self._stage2_region_proposal_network.reg_loss(y_pred = rpn_box_deltas, gt_rpn_map = gt_rpn_map)
            detector_class_loss = self._stage3_detector_network.class_loss(y_pred = detector_classes, y_true = gt_classes, from_logits = not self._activate_class_outputs)
            detector_reg_loss = self._stage3_detector_network.reg_loss(y_pred = detector_box_deltas, y_true = gt_box_deltas)
            self.add_loss(rpn_class_loss)
            self.add_loss(rpn_reg_loss)
            self.add_loss(detector_class_loss)
            self.add_loss(detector_reg_loss)
            self.add_metric(rpn_class_loss, name = 'rpn_class_loss')
            self.add_metric(rpn_reg_loss, name = 'rpn_reg_loss')
            self.add_metric(detector_class_loss, name = 'detector_class_loss')
            self.add_metric(detector_reg_loss, name = 'detector_reg_loss')
        else:
            #Losses cannot be computed during inference and should be ignored
            rpn_class_loss = float("inf")
            rpn_reg_loss = float("inf")
            detector_class_loss = float("inf")
            detector_reg_loss = float("inf")

        #Return outputs
        return[
            rpn_scores,
            rpn_box_deltas,
            detector_classes,
            detector_box_deltas,
            proposals,
            rpn_class_loss,
            rpn_reg_loss,
            detector_class_loss,
            detector_reg_loss
        ]

    def predict_on_batch(self, x, score_threshold):
        """
        Use this method to run inference. Overrides the default Keras
        implementation to return scored boxes.
        Parameters
        ----------
        x : List[np.ndarray]
        List of input maps, each of batch size 1:
            - Input image: (1, height_pixels, width_pixels, 3)
            - Anchor map: (1, height, width, num_anchors * 4)
            - Anchor valid map: (1, height, width, num_anchors)
        score_threshold : float
        Minimum class score for detections. Detections scoring below this value
        are discarded.
        Returns
        -------
        Dict[int, Tuple[float, float, float, float, float]]
        Scored boxes by class index. Each box is a tuple of
        (y_min, x_min, y_max, x_max, score).
        """
        _, _, detector_classes, detector_box_deltas, proposals, _, _, _, _ = super().predict_on_batch(x = x)
        scored_boxes_by_class_index = self._predictions_to_scored_boxes(
            input_img = x[0],
            classes = detector_classes,
            box_deltas = detector_box_deltas,
            proposals = proposals,
            score_threshold = score_threshold
        )
        return scored_boxes_by_class_index

    def load_imagenet_weights(self):
        """
        Load weights from Keras VGG-16 model pre-trained on ImageNet into the
        feature extractor convolutional layers as well as the two fully connected
        layers in the detector stage.
        """
        pretrained_vgg16_model = VGG16(weights = "imagenet")
        for pretrained_layer in pretrained_vgg16_model.layers:
            weights = pretrained_layer.get_weights()
            if len(weights) > 0:
                vgg16_layers = self._stage1_feature_extractor.layers + self._stage3_detector_network.layers
                used_layer = [layer for layer in vgg16_layers if layer.name == pretrained_layer.name]
                if len(used_layer):
                    print("Loading VGG-16 ImageNet weights into layer: %s" % used_layer[0].name)
                    used_layer[0].set_weights(weights)
        

    def _predictions_to_scored_boxes(self, input_img, classes, box_deltas, proposals, score_threshold):
        #Eliminate batch dimension
        input_img = np.squeeze(input_img, axis = 0)
        classes = np.squeeze(classes, axis = 0)
        box_deltas = np.squeeze(box_deltas, axis = 0)

        #Convert logits to probability distribution if using logits mode
        if not self._activate_class_outputs:
            classes = tf.nn.softmax(classes, axis = 1).numpy()

        #Convert proposal boxes -> center point and size
        proposal_anchors = np.empty(proposals.shape)
        proposal_anchors[:,0] = 0.5 * (proposals[:,0] + proposals[:,2])     #Y center
        proposal_anchors[:,1] = 0.5 * (proposals[:,1] + proposals[:,3])     #X center
        proposal_anchors[:,2:4] = proposals[:,2:4] - proposals[:,0:2]         #height, width

        #Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
        boxes_and_scores_by_class_indexes = {}
        for class_index in range(1, classes.shape[1]):      # skip class 0 (background)
            #Get the regression parameters (ty, tx, th, tw) corresponding to this
            #class, for all proposals
            box_delta_index = (class_index - 1) * 4
            box_delta_params = box_deltas[:, (box_delta_index + 0) : (box_delta_index + 4)]     #(N, 4)
            proposal_boxes_in_class = math.convert_deltas_to_boxes(box_deltas = box_delta_params,
                                                          anchors = proposal_anchors,
                                                          box_delta_means = [0.0, 0.0, 0.0, 0.0],
                                                          box_delta_stds = [0.1, 0.1, 0.2, 0.2])
            
            #Clip to image boundaries
            proposal_boxes_in_class[:, 0::2] = np.clip(proposal_boxes_in_class[:, 0::2], 0, input_img.shape[0] - 1)    #clip y1 and y2 to [0,height)
            proposal_boxes_in_class[:, 1::2] = np.clip(proposal_boxes_in_class[:, 1::2], 0, input_img.shape[1] - 1)    #clip x1 and x2 to [0,width)

            #Get the scores for this class. The class scores are returned in normalized categorical form. Each row corresponds to a class.
            scores_in_class = classes[:,class_index]

            #Keep only those scoring high enough
            high_scores_indexes = np.where(scores_in_class > score_threshold)[0]
            proposal_boxes_in_class = proposal_boxes_in_class[high_scores_indexes]
            scores_in_class = scores_in_class[high_scores_indexes]
            boxes_and_scores_by_class_indexes[class_index] = (proposal_boxes_in_class, scores_in_class)

        #Perform NMS per class
        scores_boxes_by_class_index = {}
        for class_index, (boxes, scores) in boxes_and_scores_by_class_indexes.items():
            indexes = tf.image.non_max_suppression(boxes = boxes,
                                                   scores = scores,
                                                   max_output_size = proposals.shape[0],
                                                   iou_threshold = 0.3)
            indexes = indexes.numpy()
            boxes = boxes[indexes]
            scores = np.expand_dims(scores[indexes], axis = 0)      # (N,) -> (N,1)
            scored_boxes = np.hstack([boxes, scores.T])             # (N,5), with each row: (y1, x1, y2, x2, score)
            scores_boxes_by_class_index[class_index] = scored_boxes

        return scores_boxes_by_class_index
    
    def _label_proposals(self, proposals, gt_box_class_indexes, gt_box_corners, min_background_iou_threshold, min_object_iou_threshold):
        """
        Determines which proposals generated by the RPN stage overlap with ground
        truth boxes and creates ground truth labels for the subsequent detector
        stage.
        Parameters
        ----------
        proposals : tf.Tensor
        Proposal corners, shaped (N, 4), where each corner is:
        (y_min, x_min, y_max, x_max).
        gt_box_class_idxs : tf.Tensor
        The class index for each ground truth box, shaped (M,), where M is the
        number of ground truth boxes.
        gt_box_corners: tf.Tensor
        Ground truth box corners, shaped (M, 4).
        min_background_iou_threshold : float
        Minimum IoU threshold with ground truth boxes below which proposals are
        ignored entirely. Proposals with an IoU threshold in the range
        [min_background_iou_threshold, min_object_iou_threshold) are labeled as
        background. This value can be greater than 0, which has the effect of
        selecting more difficult background examples that have some degree of
        overlap with ground truth boxes.
        min_object_iou_threshold : float
        Minimum IoU threshold for a proposal to be labeled as an object.
        Returns
        -------
        tf.Tensor, tf.Tensor, tf.Tensor
        Proposals, (N, 4), labeled as either objects or background (depending on
        IoU thresholds, some proposals can end up as neither and are excluded
        here); one-hot encoded class labels, (N, num_classes), for each proposal;
        and box delta regression targets, (N, 2, (num_classes - 1) * 4), for each
        proposal. Regression target values are present at locations [:,1,:] and
        consist of (ty, tx, th, tw) for the class that the box corresponds to.
        The entries for all other classes and the background classes should be
        ignored. A mask is written to locations [:,0,:]. For each proposal
        assigned a non-background class, there will be 4 consecutive elements
        marked with 1 indicating the corresponding regression target values are
        to be used. There are no regression targets for background proposals and
        the mask is entirely 0 for those proposals.
        """
        #Let's be crafty and create some fake proposals that match the ground
        #truth boxes exactly. This isn't strictly necessary and the model should
        #work without it but it will help training and will ensure that there are
        #always some positive examples to train on.
        proposals = tf.concat([proposals, gt_box_corners], axis = 0)

        #Compute IoU between each proposal (N,4) and each ground truth box (M,4)
        #-> (N, M)
        ious = math.tf_iou(box1 = proposals, box2 = gt_box_corners)

        #Find the best IoU for each proposal, the class of the ground truth box
        #associated with it, and the box corners
        best_ious = tf.math.reduce_max(ious, axis = 1)        #(N,) of maximum IoUs for each of the N proposals
        box_indexes = tf.math.argmax(ious, axis = 1)          #(N,) of ground truth box index for each proposal
        gt_box_class_indexes = tf.gather(gt_box_class_indexes, indices = box_indexes)   #(N,) of class indices of highest-IoU box for each proposal
        gt_box_corners = tf.gather(gt_box_corners, indices = box_indexes)               #(N,4) of box corners of highest-IoU box for each proposal

        #Remove all proposals whose best IoU is less than the minimum threshold
        #for a negative (background) sample. We also check for IoUs > 0 because
        #due to earlier clipping, we may get invalid 0-area proposals.
        indexes = tf.where(best_ious >= min_background_iou_threshold)       #keep proposals w/ sufficiently high IoU
        proposals = tf.gather_nd(proposals, indices = indexes)
        best_ious = tf.gather_nd(best_ious, indices = indexes)
        gt_box_class_indexes = tf.gather_nd(gt_box_class_indexes, indices = indexes)
        gt_box_corners = tf.gather_nd(gt_box_corners, indices = indexes)

        #IoUs less than min_object_iou_threshold will be labeled as background
        mask = tf.cast(best_ious >= min_object_iou_threshold, dtype = gt_box_class_indexes.dtype)   #(N,), with 0 wherever best_iou < threshold, else 1
        gt_box_class_indexes = gt_box_class_indexes * mask

        #One-hot encode class labels
        num_classes = self._num_classes
        gt_classes = tf.one_hot(indices = gt_box_class_indexes, depth = num_classes)

        #Convert proposals and ground truth boxes into "anchor" format (center
        #points and side lengths). For the detector stage, the proposals serve as
        #the anchors relative to which the final box predictions will be
        #regressed.
        proposal_centers = 0.5 * (proposals[:,0:2] + proposals[:,2:4])          #center_y, center_x
        proposal_sides = proposals[:,2:4] - proposals[:,0:2]                    #height, width
        gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])  #center_y, center_x
        gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]            #height, width

        #Compute regression targets (ty, tx, th, tw) for each proposal based on
        #the best box selected
        detector_box_delta_means = tf.constant([0, 0, 0, 0], dtype = tf.float32)
        detector_box_delta_stds = tf.constant([0.1, 0.1, 0.2, 0.2], dtype = tf.float32)
        #ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
        ty_tx = (gt_box_centers - proposal_centers) / proposal_sides
        #th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
        th_tw = tf.math.log(gt_box_sides / proposal_sides)
        #(N,4) box delta regression targets tensor
        box_delta_targets = tf.concat([ty_tx, th_tw], axis = 1)
        # mean and standard deviation adjustment
        box_delta_targets = (box_delta_targets - detector_box_delta_means) / detector_box_delta_stds

        #Convert regression targets into a map of shape (N,2,4*(C-1)) where C is
        #the number of classes and [:,0,:] specifies a mask for the corresponding
        #target components at [:,1,:]. Targets are ordered (ty, tx, th, tw).
        #Background class 0 is not present at all.
        gt_box_delta_masks = tf.repeat(gt_classes, repeats = 4, axis = 1)[:,4:]                 #create masks using interleaved repetition, remembering to discard class 0
        gt_box_delta_values = tf.tile(box_delta_targets, multiples = [1, num_classes - 1])      #populate regression targets with straightforward repetition of each row (only those columns corresponding to class will be masked on)
        gt_box_delta_masks = tf.expand_dims(gt_box_delta_masks, axis = 0)                       #(N,4*(C-1)) -> (1,N,4*(C-1))
        gt_box_delta_values = tf.expand_dims(gt_box_delta_values, axis = 0)                     #(N,4*(C-1)) -> (1,N,4*(C-1))
        gt_box_deltas = tf.concat([gt_box_delta_masks, gt_box_delta_values], axis = 0)          #(2,N,4*(C-1))
        gt_box_deltas = tf.transpose(gt_box_deltas, perm = [1, 0, 2])                           #(N,2,4*(C-1))

        return proposals, gt_classes, gt_box_deltas

    def _sample_proposals(self, proposals, gt_classes, gt_box_deltas, max_proposals, pos_fraction):
        if max_proposals <= 0:
            return proposals, gt_classes, gt_box_deltas
        
        #Get positive and negative (background) proposals
        class_indices = tf.argmax(gt_classes, axis = 1)                     #(N,num_classes) -> (N,), where each element is the class index (highest score from its row)
        pos_indices = tf.squeeze(tf.where(class_indices > 0), axis = 1)     #(P,), tensor of P indices (the positive, non-background classes in class_indices)
        neg_indices = tf.squeeze(tf.where(class_indices <= 0), axis = 1)    #(N,), tensor of N indices (the negative, background classes in class_indices)
        num_pos_proposals = tf.size(pos_indices)
        num_neg_proposals = tf.size(neg_indices)

        #Select positive and negative samples, if there are enough. Note that the
        #number of positive samples can be either the positive fraction of the
        #*actual* number of proposals *or* the *desired* number (max_proposals).
        #In practice, these yield virtually identical results but the latter
        #method will yield slightly more positive samples in the rare cases when
        #the number of proposals is below the desired number. Here, we use the
        #former method but others, such as Yun Chen, use the latter. To implement
        #it, replace num_samples with max_proposals in the line that computes
        #num_positive_samples. I am not sure what the original Faster R-CNN
        #implementation does.
        num_samples = tf.minimum(max_proposals, tf.size(class_indices))
        num_pos_samples = tf.minimum(tf.cast(tf.math.round(tf.cast(num_samples, dtype = float) * pos_fraction), dtype = num_samples.dtype), num_pos_proposals)
        num_neg_samples = tf.minimum(num_samples - num_pos_samples, num_neg_proposals)

        #Sample randomly
        pos_sample_indices = tf.random.shuffle(pos_indices)[:num_pos_samples]
        neg_sample_indices = tf.random.shuffle(neg_indices)[:num_neg_samples]
        indices = tf.concat([pos_sample_indices, neg_sample_indices], axis = 0)

        #My initial PyTorch version was careful to return empty tensors if there
        #were no positive samples or no negative samples. Because TF2/Keras is awful
        #and tf.cond doesn't work due to some incompatibility between tf.function
        #and KerasTensor, we always return the proposals even if there are no
        #negative samples among them. Ths occurs very rarely. Positive samples are
        #guaranteed to exist because _label_proposals inserts the ground truth boxes
        #as fake proposals to boost learning.
        """
        no_samples = tf.math.logical_or(tf.math.less_equal(num_pos_samples, 0), tf.math.less_equal(num_neg_samples, 0))

        #Return (if we have any samples)
        proposals = tf.cond(no_samples,
                            true_fn = lambda: tf.zeros(shape = (0, 4), dtype = proposals.dtype),    #empty proposals tensor if no samples
                            false_fn = lambda: tf.gather(proposals, indices = indices))             #gather samples
        gt_classes = tf.cond(no_samples,
                            true_fn = lambda: tf.zeros(shape = (0, tf.shape(gt_classes)[1]), dtype = gt_classes.dtype),    #empty proposals tensor if no samples
                            false_fn = lambda: tf.gather(gt_classes, indices = indices))             #gather samples
        gt_box_deltas = tf.cond(no_samples,
                            true_fn = lambda: tf.zeros(shape = (0, tf.shape(gt_box_deltas)[1]), dtype = gt_box_deltas.dtype),    #empty proposals tensor if no samples
                            false_fn = lambda: tf.gather(gt_box_deltas, indices = indices))             #gather samples
        """

        return tf.gather(proposals, indices = indices), tf.gather(gt_classes, indices = indices), tf.gather(gt_box_deltas, indices = indices)
        


        

















            


        

        
