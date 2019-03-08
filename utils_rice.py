import numpy as np
from matplotlib import pyplot as plt
import math, sys, pdb, os
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Add, Conv2DTranspose, Flatten, Dense, Conv1D, AveragePooling2D, LeakyReLU, PReLU, GlobalAveragePooling2D
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.models import Model

##################################################################################
###########################      Seed Normalization     ##########################
##################################################################################

def normalizeDataWholeSeed(data,normalization_type='max'):
    
    if normalization_type == 'max':
        for idx in range(data.shape[0]):
            data[idx,:,:,:] = data[idx,:,:,:]/np.max(abs(data[idx,:,:,:]))
            
    elif normalization_type == 'l2norm':
        from numpy import linalg as LA
        for idx in range(data.shape[0]):
            data[idx,:,:,:] = data[idx,:,:,:]/LA.norm(data[idx,:,:,:]) # L2-norm by default        
        
    return data

##################################################################################
################################     hparams    ##################################
##################################################################################

# Make the param list
def make_hyperparam_string(USE_DATA_AUG, learning_rate_base, batch_size, kernel_size, dropout_rate, num_training,
                           num_nodes_fc, activation_type):
    hparam = ""

    # Hyper-parameters
    if USE_DATA_AUG:
        hparam += "AUG_"

    hparam += str(num_nodes_fc) + "nodes_" + str(learning_rate_base) + "lr_" + str(batch_size) + "batch_" + str(
        kernel_size) + "kernel_" + str(dropout_rate) + "drop_" + str(
        num_training) + "train_" + activation_type

    return hparam

##################################################################################
##########################      Confusion Matrix     #############################
##################################################################################

# Print and plot a confusion matrix
# Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.clim(0,sum(cm[0,:]))
    plt.xlabel('Predicted label')

##################################################################################
########################      Classification Accuracy     ########################
##################################################################################

def top_K_classification_accuracy(y_predicted, y_true, K=1):

    num_samples = y_predicted.shape[0]
    num_classes = y_predicted.shape[1]

    if K > num_classes:
        sys.exit(1)

    temp = np.zeros((num_samples,))

    for idx in range(num_samples):
        curr_predicted = np.argsort(y_predicted[idx,:])
        curr_predicted = curr_predicted[::-1] # descending

        if y_true[idx] in curr_predicted[:K]:
            temp[idx] = 1

    return 100.0 * np.sum(temp)/num_samples


##################################################################################
################################      ResNet     #################################
##################################################################################

def conv2D_ResNet(x, kernel_size, activation_type, dropout_rate, num_filters_first_conv1D):

    x_orig = x

    # Batch norm
    x = BatchNormalization()(x)

    # 1x1 Conv2D
    x = Conv2D(num_filters_first_conv1D, kernel_size=1, activation=None, use_bias=False, padding='same',
               kernel_initializer='truncated_normal')(x)

    # Activation
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)
    else:
        x = Activation(activation_type)(x)

    x = BatchNormalization()(x)

    # 3x3 Conv2D
    x = Conv2D(num_filters_first_conv1D, kernel_size, activation=None, use_bias=True, padding='same',
               kernel_initializer='truncated_normal')(x)

    # Activation
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)
    else:
        x = Activation(activation_type)(x)

    x = BatchNormalization()(x)

    # 1x1 Conv2D
    x = Conv2D(num_filters_first_conv1D*4, kernel_size=1, activation=None, use_bias=False, padding='same',
               kernel_initializer='truncated_normal')(x)

    # Skip connection
    if int(x.shape[3]) != int(x_orig.shape[3]):
        x_orig = Conv2D(int(x.shape[3]), kernel_size=1, activation=None, use_bias=False, padding='same',
               kernel_initializer='truncated_normal')(x_orig)

    # Activation
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)
    else:
        x = Activation(activation_type)(x)

    x = Add()([x, x_orig])

    # Dropout
    return Dropout(dropout_rate)(x)


def createBlock_ResNet2D(x, num_layers, kernel_size, activation_type, dropout_rate, num_filters_first_conv1D):

    for idx_layer in range(num_layers):

        x = conv2D_ResNet(x, kernel_size, activation_type, dropout_rate, num_filters_first_conv1D)

    return x

# growth_rate: number of filters for each normal convolution ('k' in the paper)
def ResNet2D_classifier(data_num_rows, data_num_cols, num_classes, kernel_size=3, num_layers_each_block=[6, 12, 24, 16],
                        num_chan_per_block = [64,128,256,512], activation_type='swish', dropout_rate=0.0, num_input_chans=1, num_nodes_fc=64):

    input_data = Input(shape=(data_num_rows, data_num_cols, num_input_chans))

    # Input layer: Conv2D -> activation
    x = Conv2D(num_chan_per_block[0], kernel_size, activation=None, use_bias=True, padding='same',
               kernel_initializer='truncated_normal')(input_data)

    # Activation
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)
    else:
        x = Activation(activation_type)(x)

    #  Blocks & Downsampling Layers
    for idx_block in range(len(num_layers_each_block)):
        x = createBlock_ResNet2D(x, num_layers_each_block[idx_block], kernel_size, activation_type, dropout_rate,
                                 num_chan_per_block[idx_block])

        x = BatchNormalization()(x)

        if idx_block != len(num_layers_each_block)-1:
            x = Conv2D(num_chan_per_block[idx_block]*2, kernel_size, strides = 2, activation=None, use_bias=True, padding='valid',
                   kernel_initializer='truncated_normal')(x)
        else:
            x = GlobalAveragePooling2D()(x)

        x = Dropout(dropout_rate)(x)

    # Output layer
    x = BatchNormalization()(x)
    x = Dense(units=num_nodes_fc, activation=None, kernel_initializer='truncated_normal')(x)

    # Activation
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)
    else:
        x = Activation(activation_type)(x)

    x = BatchNormalization()(x)
    output_data = Dense(units=num_classes, activation='softmax', kernel_initializer='truncated_normal')(x)

    return Model(inputs=input_data, outputs=output_data)