#import Libraries
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from conf import myConfig as config
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import tensorflow.nn as nn
import tensorflow.train as tftrain
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from numpy import *
import random
import argparse
import os
import datetime
from keras.layers import LeakyReLU

# import the libraries
import os
import numpy as np
import cv2
from pathlib import Path




from tensorflow.keras.layers import LeakyReLU as PReLU

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', dest='dataPath', type=str, default='./BSD68', help='testDataPath')
parser.add_argument('--weightsPath', dest='weightsPath', type=str, default='./training_checkpoints/ckpt_{0:03d}', help='pathOfTrainedCNN')
parser.add_argument('--epoch', dest='epoch', type=int, default=52)
parser.add_argument('--noise', dest='noise_level_img', type=int, default=25)
args = parser.parse_args()

# Combined loss function

def my_Hfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[0]], [[1]]],
            [[[-2]], [[0]], [[2]]],
            [[[-1]], [[0]], [[1]]]
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_Vfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[-2]], [[-1]]],
            [[[0]], [[0]], [[0]]],
            [[[1]], [[2]], [[1]]]
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_D1filter(shape, dtype=None):
    f = np.array([
            [[[-2]], [[-1]], [[0]]],
            [[[-1]], [[0]], [[1]]],
            [[[0]], [[1]], [[2]]]
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_D2filter(shape, dtype=None):
    f = np.array([
            [[[0]], [[1]], [[2]]],
            [[[-1]], [[0]], [[1]]],
            [[[-2]], [[-1]], [[0]]]
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res

# Load the trained model
weights_path_resolved = args.weightsPath.format(args.epoch)
if not os.path.exists(weights_path_resolved):
    raise FileNotFoundError(f"Checkpoint file not found at {weights_path_resolved}")

nmodel_PROPOSED = load_model(weights_path_resolved, custom_objects={'custom_loss': custom_loss, 'my_Hfilter': my_Hfilter,'my_Vfilter': my_Vfilter,'my_D1filter': 	my_D1filter,'my_D2filter': my_D2filter})
print('Trained Model is loaded')

# Create array of test images
p = Path(args.dataPath)
listPaths = list(p.glob('./*.png'))
imgTestArray = [cv2.imread(str(path), 0) for path in listPaths]
imgTestArray = np.array(imgTestArray, dtype=object) / 255.0

length = len(imgTestArray)

# Create result directories
base_path = f"./Test_Results/Synthetic/{args.epoch}/{args.noise_level_img}/"
folder_path_original = os.path.join(base_path, "Original/")
folder_path_noisy = os.path.join(base_path, "Noisy/")
folder_path_proposed = os.path.join(base_path, "Proposed/")
os.makedirs(folder_path_original, exist_ok=True)
os.makedirs(folder_path_noisy, exist_ok=True)
os.makedirs(folder_path_proposed, exist_ok=True)

metric_path_proposed = os.path.join(base_path, "metric.txt")
with open(metric_path_proposed, 'w') as file_m:
    file_m.write(f'Metric: {args.dataPath}; Epoch: {args.epoch}; Noise Level: {args.noise_level_img}\n')

# Process images and calculate metrics
sumPSNR = 0
sumSSIM = 0
psnr_val = np.empty(length)
ssim_val = np.empty(length)

for i in range(length):
    np.random.seed(seed=0)  # for reproducibility
    img1 = imgTestArray[i]
    f = img1 + np.random.normal(0, args.noise_level_img / 255.0, img1.shape)
    z = np.squeeze(nmodel_PROPOSED.predict(np.expand_dims(f, axis=0)))

    # Resize the model output to match the input dimensions
    z_resized = cv2.resize(z, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Save images
    cv2.imwrite(os.path.join(folder_path_original, f"{i + 1}_Original.png"), (255.0 * img1).astype(np.uint8))
    cv2.imwrite(os.path.join(folder_path_noisy, f"{i + 1}_Noisy.png"), (255.0 * f).astype(np.uint8))
    cv2.imwrite(os.path.join(folder_path_proposed, f"{i + 1}_Proposed.png"), (255.0 * z_resized).astype(np.uint8))

    # Compute metrics using resized output
    psnr_val[i] = psnr(img1, z_resized)
    ssim_val[i] = ssim(img1, z_resized, channel_axis=-1)

    with open(metric_path_proposed, 'a') as file_m:
        file_m.write(f'PSNR of image {i + 1} is {psnr_val[i]}\n')
        file_m.write(f'SSIM of image {i + 1} is {ssim_val[i]}\n')

    sumPSNR += psnr_val[i]
    sumSSIM += ssim_val[i]

# Compute average metrics
avgPSNR = sumPSNR / length
avgSSIM = sumSSIM / length

with open(metric_path_proposed, 'a') as file_m:
    file_m.write(f'\n\navgPSNR on dataset = {avgPSNR}\n')
    file_m.write(f'avgSSIM on dataset = {avgSSIM}\n\n')

print(f"Processing complete. Average PSNR: {avgPSNR}, Average SSIM: {avgSSIM}")

