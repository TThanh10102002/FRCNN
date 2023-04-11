import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Dropout, Flatten, Lambda, TimeDistributed
from keras import backend as K
from keras.regularizers import L2
from keras.initializers import random_normal
from keras.losses import CategoricalCrossentropy
from tensorflow import math


from .ROI_pooling import RoIPoolingLayer


class DetectorNet(Model):
    def __init__(self, num_classes, custom_roi_pool, activate_class_outputs, l2, dropout):
        super().__init__()

        self._num_classes = num_classes
        self._activate_class_outputs = activate_class_outputs
        self._dropout = dropout

        regularizer = L2(l2)
        class_initializer = random_normal(mean = 0.0, stddev = 0.01)
        regressor_initializer = random_normal(mean = 0.0, stddev = 0.001)

        self._roi_pool = RoIPoolingLayer(pool_size=7, name = 'custom_roi_pool') if custom_roi_pool else None

        self._flatten = TimeDistributed(Flatten())
        self._fc1 = TimeDistributed(name = "fc1", layer = Dense(units = 4096, activation = 'relu', kernel_regularizer = regularizer))
        self._dropout1 = TimeDistributed(Dropout(dropout))
        self._fc2 = TimeDistributed(name = "fc2", layer = Dense(units = 4096, activation = 'relu', kernel_regularizer = regularizer))
        self._dropout2 = TimeDistributed(Dropout(dropout))

        class_activation = "softmax" if activate_class_outputs else None
        self._classifier = TimeDistributed(name = "classifier_class", layer = Dense(units = num_classes, activation=class_activation, kernel_initializer = class_initializer))
        
        self._regressor = TimeDistributed(name = "classifier_boxes", layer = Dense(units = 4 * (num_classes - 1), activation = "linear", kernel_initializer = regressor_initializer))

    def call(self, inputs, training):
        input_img = inputs[0]
        feature_map = inputs[1]
        proposals = inputs[2]
        assert len(feature_map.shape) == 4

        if self._roi_pool:
            proposals = tf.cast(proposals, dtype = tf.int32)
            map_dim = tf.shape(feature_map)[1:3]
            map_lim = tf.tile(map_dim, multiples = [2]) - 1
            roi_corners = tf.minimum(proposals // 16, map_lim)
            roi_corners = tf.maximum(roi_corners, 0)
            roi_dim = roi_corners[:, 2:4] - roi_corners[:, 0:2] + 1
            rois = tf.concat([roi_corners[:, 0:2], roi_dim], axis = 1)
            rois = tf.expand_dims(rois, axis = 0)
            pool = RoIPoolingLayer(pool_size=7, name = "roi_pool")([feature_map, rois])
        else:
            img_height = tf.shape(input_img)[1]
            img_width = tf.shape(input_img)[2]
            rois = proposals / [img_height, img_width, img_height, img_width]

            num_rois = tf.shape(rois)[0];
            region = tf.image.crop_and_resize(image = feature_map, boxes = rois, box_indices = tf.zeros(num_rois, dtype = tf.int32), crop_size = [14, 14])
            pool = tf.nn.max_pool(region, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
            pool = tf.expand_dims(pool, axis = 0)
            
        flatten_layer = self._flatten(pool)
        if training and self._dropout != 0:
            fc1 = self._fc1(flatten_layer)
            dropout1 = self._dropout1(fc1)
            fc2 = self._fc2(dropout1)
            dropout2 = self._dropout2(fc2)
            out = dropout2
        else:
            fc1 = self._fc1(flatten_layer)
            fc2 = self._fc2(fc1)
            out = fc2
        class_activation = 'softmax' if self._activate_class_outputs else None
        classes = self._classifier(out)
        box_deltas = self._regressor(out)

        return [classes, box_deltas]

    @staticmethod
    def class_loss(y_pred, y_true, from_logits):
        scale_factor = 1.0
        N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon()
        if from_logits:
            return scale_factor * math.reduce_sum(CategoricalCrossentropy(from_logits = True)(y_true, y_pred)) / N
        else:
            return scale_factor * math.reduce_sum(CategoricalCrossentropy()(y_true, y_pred)) / N

    @staticmethod
    def reg_loss(y_pred, y_true):
        scale_factor = 1.0
        sigma = 1.0
        squared_sigma = sigma * sigma

        y_mask = y_true[:, :, 0, :]
        y_true_targets = y_true[:, :, 1, :]

        x = y_true_targets - y_pred
        abs_x = tf.math.abs(x)
        is_negative_branch = tf.stop_gradient(tf.cast(tf.less(abs_x, 1.0 / squared_sigma), dtype = tf.float32))
        R_negative_branch = 0.5 * x * x * squared_sigma
        R_positive_branch = abs_x - 0.5 / squared_sigma
        losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

        N = tf.cast(tf.shape(y_true)[1], dtype = tf.float32) + K.epsilon()
        loss_terms = y_mask * losses
        return scale_factor * math.reduce_sum(loss_terms) / N









            

