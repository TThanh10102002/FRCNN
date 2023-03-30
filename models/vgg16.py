import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import glorot_normal

class FeatureExtractor(tf.keras.Model):
    def __init__(self, l2 = 0):
        super().__init__()
        
        initial_weights = glorot_norrmal()
        regularizer = tf.keras.regularizers.l2(l2)
        input_shape = (None, None, 3)

        self._block1_conv1 = Conv2D(input_shape = input_shape, filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, trainable = False, name = 'block1_conv1')
        self._block1_conv2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, trainable = False, name = 'block1_conv2')
        self._block1_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block2_conv1 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, trainable = False, name = 'block2_conv1')
        self._block2_conv2 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, trainable = False, name = 'block2_conv2')
        self._block2_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block3_conv1 = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block3_conv1')
        self._block3_conv2 = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block3_conv2')
        self._block3_conv3 = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block3_conv3')
        self._block3_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block4_conv1 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block4_conv1')
        self._block4_conv2 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block4_conv2')
        self._block4_conv3 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block4_conv3')
        self._block4_maxpool = MaxPooling2D(pool_size = 2, strides = 2)

        self._block5_conv1 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block5_conv1')
        self._block5_conv2 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block5_conv2')
        self._block5_conv3 = Conv2D(filters = 512, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = initial_weights, kernel_regularizer = regularizer, name = 'block5_conv3')

    def call(self, input_img):
        y = self._block1_conv1(input_img)
        y = self._block1_conv2(y)
        y = self._block1_maxpool(y)

        y = self._block2_conv1(y)
        y = self._block2_conv2(y)
        y = self._block2_maxpool(y)

        y = self._block3_conv1(y)
        y = self._block3_conv2(y)
        y = self._block3_conv3(y)
        y = self._block3_maxpool(y)

        y = self._block4_conv1(y)
        y = self._block4_conv2(y)
        y = self._block4_conv3(y)
        y = self._block4_maxpool(y)

        y = self._block5_conv1(y)
        y = self._block5_conv2(y)
        y = self._block5_conv3(y)

        return y
    
