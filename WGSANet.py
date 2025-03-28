import tensorflow as tf
from tensorflow.keras.layers import Layer  # Import the Layer class
from tensorflow.keras.layers import Layer, BatchNormalization, Activation, Add
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, LeakyReLU, Concatenate, Input
import pywt  # Import pywt for wavelet transforms
from tensorflow.keras import layers


# import the libraries
import os
import numpy as np
import cv2
from conf import myConfig as config
from pathlib import Path
from conf import myConfig as config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import tensorflow.data as tfdata
import tensorflow.image as tfimage
import tensorflow.nn as nn
import tensorflow.train as tftrain
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from numpy import *
import random
import os
from glob import glob
import datetime
import argparse
import PIL
import tensorflow.keras.backend as K
from PIL import Image
from PIL import Image, ImageOps
import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
# import the libraries

import numpy as np
import cv2
#shape = (3, 3, 3, 1)
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, Activation, Add, AveragePooling2D, MaxPooling2D, UpSampling2D, Multiply #Import AveragePooling2D, MaxPooling2D, UpSampling2D, Multiply here
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, LeakyReLU, Concatenate, Input
import pywt  
from tensorflow.keras import layers



# Custom Kernels for Multi-Directional Gradient Estimation
def my_Hfilter(shape, dtype=None):
    f = np.array([
        [[[-1], [0], [1]], [[-1], [0], [1]], [[-1], [0], [1]]],   # Red channel kernel
        [[[-2], [0], [2]], [[-2], [0], [2]], [[-2], [0], [2]]],   # Green channel kernel
        [[[-1], [0], [1]], [[-1], [0], [1]], [[-1], [0], [1]]]    # Blue channel kernel
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_Vfilter(shape, dtype=None):
    f = np.array([
        [[[-1], [-2], [-1]], [[-1], [-2], [-1]], [[-1], [-2], [-1]]],   # Red channel kernel
        [[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]],            # Green channel kernel
        [[[1], [2], [1]], [[1], [2], [1]], [[1], [2], [1]]]             # Blue channel kernel
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_D1filter(shape, dtype=None):
    f = np.array([
        [[[-2], [-1], [0]], [[-2], [-1], [0]], [[-2], [-1], [0]]],      # Red channel kernel
        [[[-1], [0], [1]], [[-1], [0], [1]], [[-1], [0], [1]]],         # Green channel kernel
        [[[0], [1], [2]], [[0], [1], [2]], [[0], [1], [2]]]             # Blue channel kernel
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_D2filter(shape, dtype=None):
    f = np.array([
        [[[0], [1], [2]], [[0], [1], [2]], [[0], [1], [2]]],            # Red channel kernel
        [[[-1], [0], [1]], [[-1], [0], [1]], [[-1], [0], [1]]],         # Green channel kernel
        [[[-2], [-1], [0]], [[-2], [-1], [0]], [[-2], [-1], [0]]]       # Blue channel kernel
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')
# Wrap tf.sqrt in a custom Keras layer
class SqrtLayer(Layer):
    def call(self, inputs):
        return tf.sqrt(inputs)


# Multi-Directional Gradient Magnitude Estimator
def multi_directional_gradient_magnitude_estimator(input_tensor):
    h_filter = tf.keras.layers.Conv2D(1, (3, 3), padding='same', kernel_initializer=my_Hfilter, use_bias=False)(input_tensor)
    v_filter = tf.keras.layers.Conv2D(1, (3, 3), padding='same', kernel_initializer=my_Vfilter, use_bias=False)(input_tensor)
    d1_filter = tf.keras.layers.Conv2D(1, (3, 3), padding='same', kernel_initializer=my_D1filter, use_bias=False)(input_tensor)
    d2_filter = tf.keras.layers.Conv2D(1, (3, 3), padding='same', kernel_initializer=my_D2filter, use_bias=False)(input_tensor)

    # Combine the gradient magnitude in all directions using the custom SqrtLayer
    combined_gradient = SqrtLayer()(h_filter*2 + v_filter*2 + d1_filter*2 + d2_filter*2)  
    return combined_gradient



def adaptive_residual_attention_module(input_tensor):
    """
    Implements the Adaptive Residual Attention Module (ARAM) with the specified operations.

    Args:
        input_tensor: Input feature map (Tensor).

    Returns:
        Output tensor after applying ARAM.
    """
    # Combined pooling: Average Pooling + Top-K Max Pooling
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
    combined_pool = Add()([avg_pool, max_pool])

    # Convolutional layers with Leaky ReLU activations
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(combined_pool)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=2, padding='same', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=3, padding='same', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)
 
    x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=2, padding='same', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Upsample the attention map to match the input tensor shape
    x = UpSampling2D(size=(2, 2))(x)  

    # Sigmoid activation to generate the attention map
    attention_map = Activation('sigmoid')(x)

    # Multiplicative operation with the input tensor
    output_tensor = Multiply()([input_tensor, attention_map])

    return output_tensor


# Texture Refinement Module
class TextureRefinementModule(tf.keras.layers.Layer):
    def _init_(self, filters, input_channels):
        super(TextureRefinementModule, self)._init_()
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.grad_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.sigmoid = tf.keras.activations.sigmoid
        self.branch1_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.final_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.channel_adjust = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')  


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)

        branch1 = x
        branch2 = x

        normalized_gradient = self.grad_conv(inputs)
        normalized_gradient = self.sigmoid(normalized_gradient)

        branch1 = self.branch1_conv(branch1)
        branch1 = self.sigmoid(branch1)
        branch1 = branch1 * normalized_gradient

        add_op = branch1 + branch2

        refined = self.final_conv(add_op)
        refined = self.leaky_relu(refined)

        refined = self.channel_adjust(refined)

        output = tf.keras.layers.Multiply()([inputs, refined])  # Applying element-wise multiplication

        return output
# ==== Wavelet Domain Denoising Pipeline ====
class DWTLayer(tf.keras.layers.Layer):
    def _init_(self, wavelet='haar', **kwargs):
        super(DWTLayer, self)._init_(**kwargs)
        self.wavelet = wavelet

    def call(self, inputs):
        # Modify Tout to a flat tuple of DTypes
        LL, LH, HL, HH = tf.py_function(
            func=lambda x: pywt.dwt2(x.numpy(), self.wavelet, mode='periodization'),
            inp=[inputs],
            # Specify the output types as a flat tuple
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
        )
        # Get input shape dynamically using tf.shape
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        # Calculate output shapes using tf.cast and floor division
        output_height = tf.cast(tf.math.floor(tf.cast(height, tf.float32) / 2), tf.int32)
        output_width = tf.cast(tf.math.floor(tf.cast(width, tf.float32) / 2), tf.int32)

        # Set the shape of the output tensors dynamically using tf.ensure_shape
        # Instead of using tf.ensure_shape, use tf.reshape
        LL = tf.reshape(LL, [batch_size, output_height, output_width, channels])  
        LH = tf.reshape(LH, [batch_size, output_height, output_width, channels])  
        HL = tf.reshape(HL, [batch_size, output_height, output_width, channels])  
        HH = tf.reshape(HH, [batch_size, output_height, output_width, channels])  
        
        return LL, LH, HL, HH


class IDWTLayer(tf.keras.layers.Layer):
    def _init_(self, wavelet='haar', **kwargs):
        super(IDWTLayer, self)._init_(**kwargs)
        self.wavelet = wavelet

    def call(self, inputs):
        LL, LH, HL, HH = inputs
        
        # Wrap pywt.idwt2 in tf.py_function
        reconstructed = tf.py_function(
            func=lambda LL, LH, HL, HH: pywt.idwt2((LL.numpy(), (LH.numpy(), HL.numpy(), HH.numpy())), self.wavelet, mode='periodization'),
            inp=[LL, LH, HL, HH],  # Pass TensorFlow tensors
            Tout=tf.float32,
        )
        
        # Get the shape of LL dynamically
        input_shape = tf.shape(LL)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
        # Calculate output shape using tf.cast and multiplication
        output_height = tf.cast(height * 2, tf.int32)
        output_width = tf.cast(width * 2, tf.int32)

        # Set the shape of the reconstructed tensor dynamically
        # Instead of using set_shape, use tf.reshape
        reconstructed = tf.reshape(reconstructed, [batch_size, output_height, output_width, channels]) 
        
        return reconstructed


class MSAAB(tf.keras.layers.Layer):
    def _init_(self, filters, **kwargs):
        super(MSAAB, self)._init_(**kwargs)
        self.conv1_5x5 = Conv2D(filters, (5, 5), padding='same')
        self.conv2_5x5 = Conv2D(filters, (5, 5), padding='same')
        self.conv1_3x3 = Conv2D(filters, (3, 3), padding='same')
        self.conv2_3x3 = Conv2D(filters, (3, 3), padding='same')
        self.conv1_2x2 = Conv2D(filters, (2, 2), padding='same')
        self.conv2_2x2 = Conv2D(filters, (2, 2), padding='same')
        self.leaky_relu = LeakyReLU()
        self.dilated_conv1 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')
        self.dilated_conv2 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')
        self.dilated_conv3 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')
        # Initialize channel_adjust in _init_ 
        self.channel_adjust = None  

    def build(self, input_shape):
        # Create channel_adjust here with the input shape
        self.channel_adjust = tf.keras.layers.Conv2D(input_shape[-1], (1, 1), padding='same')
        super(MSAAB, self).build(input_shape) #must be at the end

    def call(self, inputs):
        x1 = self.leaky_relu(self.conv2_5x5(self.leaky_relu(self.conv1_5x5(inputs))))
        x2 = self.leaky_relu(self.conv2_3x3(self.leaky_relu(self.conv1_3x3(inputs))))
        x3 = self.leaky_relu(self.conv2_2x2(self.leaky_relu(self.conv1_2x2(inputs))))
        added = x1 + x2 + x3
        x = self.leaky_relu(self.dilated_conv1(added))
        x = self.leaky_relu(self.dilated_conv2(x))
        x = self.leaky_relu(self.dilated_conv3(x))
        
        # Now you can use channel_adjust
        alpha = tf.keras.activations.sigmoid(self.channel_adjust(x))  
        return alpha * inputs


class DWT_MSAAB_IDWT_Block(tf.keras.layers.Layer):
    def _init_(self, filters, wavelet='haar', **kwargs):
        super(DWT_MSAAB_IDWT_Block, self)._init_(**kwargs)
        self.dwt = DWTLayer(wavelet=wavelet)
        self.msaab = MSAAB(filters)  # Single MSAAB instance
        self.conv_bn_swish = tf.keras.Sequential([
            Conv2D(filters, (3, 3), padding='same'),
            BatchNormalization(),
            Activation(tf.keras.activations.swish),
        ])
        self.idwt = IDWTLayer(wavelet=wavelet)

    def call(self, inputs):
        LL, LH, HL, HH = self.dwt(inputs)
        
        # Concatenate high-frequency sub-bands
        high_freq_concat = tf.concat([LH, HL, HH], axis=-1)  
        
        # Process concatenated high-frequency sub-bands with MSAAB
        processed_high_freq = self.msaab(high_freq_concat)  
        
        # Concatenate LL with processed high-frequency for further processing
        LL_expanded = tf.concat([LL, processed_high_freq], axis=-1)  
        processed_LL = self.conv_bn_swish(LL_expanded)
        
        # Ensure processed_high_freq has the same number of channels as LH, HL, HH
        processed_high_freq = tf.keras.layers.Conv2D(LH.shape[-1], (1, 1), padding='same')(processed_high_freq) 
        
        # Reconstruct using IDWT with processed sub-bands
        output = self.idwt([processed_LL, processed_high_freq, HL, HH])  # Pass processed_high_freq
        return output
# Complete Model Pipeline
def build_model(input_shape=(None, None, 3)):
    input_tensor = Input(shape=input_shape)

    # Branch 1: Conv2D + LeakyReLU
    branch1 = Conv2D(64, (3, 3), padding='same')(input_tensor)
    branch1 = LeakyReLU(alpha=0.2)(branch1)

    # Branch 2: Multi-Directional Gradient Magnitude Estimator
    branch2 = multi_directional_gradient_magnitude_estimator(input_tensor)

    # Concatenate Branches
    concatenated = Concatenate()([branch1, branch2])

    # Intermediate Feature Fusion Module
    x = Conv2D(64, (3, 3), padding='same')(concatenated)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Split into Two Branches
    branch1_attention = adaptive_residual_attention_module(x)
    branch2_refinement = TextureRefinementModule(filters=64, input_channels=3)(x)

    # Concatenate the Outputs
    feature_fusion = Concatenate()([branch1_attention, branch2_refinement])

#wavelet block call
    x1 = DWT_MSAAB_IDWT_Block(filters=64, wavelet='haar')(input_tensor)
    #
    conv1 = layers.Conv2D(concatenated.shape[-1], (3, 3), padding='same')(feature_fusion)  
    x = layers.Add()([conv1, concatenated])

     # Add with wavelet output

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Add()([x, x1])

     # Add with Initial Input
    x = layers.Conv2D(input_tensor.shape[-1], (1, 1), padding='same')(x)  
    x = layers.Add()([x, input_tensor])
        # Final Refinement Layers
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    final_output = layers.Conv2D(1, (1, 1), padding='same')(x)

    # Define Model
    model = tf.keras.Model(inputs=input_tensor, outputs=final_output)
    return model

# Create Model
mdl = build_model()
mdl.summary()

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt = AdamW(learning_rate=0.001, weight_decay=0.01)
def custom_loss(y_true,y_pred):
    diff=abs(y_true-y_pred)
    #l1=K.sum(diff)/(config.batch_size)
    l1=(diff)/(config.batch_size)
    return l1
mdl.compile(loss=custom_loss,optimizer=opt)
mdl.summary()

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
#cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")


def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(0,60)
        trueNoiseBatch=np.random.normal(0,noise/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield (noisyImagesBatch,trueNoiseBatch)



cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./training_checkpoints','ckpt_{epoch:03d}'), verbose=1,save_freq='epoch')
logdir = "./training_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
lr_callback = [LearningRateScheduler(lr_decay)]
# train
mdl.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=[lr_callback, cp_callback, tensorboard_callback], verbose=1)

