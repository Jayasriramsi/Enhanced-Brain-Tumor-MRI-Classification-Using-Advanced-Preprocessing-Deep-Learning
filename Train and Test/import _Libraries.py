#openCV
import cv2
#mathmetical operation
import numpy as np
#dataframe
import pandas as pd
#tensorflow for google framework of neural network
import tensorflow as tf
#another library for NN which runs on top of tensorflow for more effecient work and functionality
import keras
#directory access
import os
#data visualization 
#module provides tools and utilities for working with images
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import backend as k
#sequential model
#enabling the creation and manipulation of Keras models
from tensorflow.keras.models import Model
#model import
from tensorflow.keras.applications import VGG19
#input details
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
#layer and regularization
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
#activation
from tensorflow.keras.optimizers import  SGD, Adam
#tensorboard
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler
#model flowchart
from tensorflow.keras.utils import plot_model
#ploting
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
#misclassification display
import matplotlib.gridspec as gridspec
from PIL import Image

#Accuracy and Confusion Matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
#ROC AUC CURVE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from sklearn.preprocessing import label_binarize #for categorical to binary conversion

#Warning
import warnings
from sklearn.exceptions import DataConversionWarning  # Import the specific warning

# Filter specific warnings
warnings.filterwarnings(action='ignore')