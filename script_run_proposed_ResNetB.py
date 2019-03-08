# Import modules
import os, pdb, timeit
import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils_rice import *
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.cm as cm
import cv2
from keras import activations
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils


##################################################################################
######################      Create and train a model      ########################
##################################################################################

def createAndTrainResNetB(params):
                                        
    ############ Extract params ############
    USE_DATA_AUG = params['USE_DATA_AUG']
    learning_rate_base = params['learning_rate_base']
    kernel_size = params['kernel_size']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    dropout_rate = params['dropout_rate']
    activation_type = params['activation_type']
    num_nodes_fc = params['num_nodes_fc']
    rice_types = params['rice_types']
    normalization_type = params['normalization_type']
    num_layers_each_block = params['num_layers_each_block']
    num_chan_per_block = params['num_chan_per_block']
    N_classes = len(rice_types)
    
    
    ############ Load data ############
    print("--------------Load Data--------------")

    # Load training data and their corresponding labels
    x_training = np.load('x.npy')
    labels_training = np.load('labels.npy')
    
    # Normalize the data
    x_training = normalizeDataWholeSeed(x_training,normalization_type=normalization_type)
    
    # Extract some information
    num_training = x_training.shape[0]
    N_spatial = x_training.shape[1:3]
    N_bands = x_training.shape[3]
    num_batch_per_epoch = int(num_training/batch_size)
    
    print('#training = %d' %(num_training))
    print('#batches per epoch = %d' %(num_batch_per_epoch))
    
    print("--------------Done--------------")
    
    
    ############ Prepare the path for saving the models/stats ############
    print("--------------Prepare a path for saving the models/stats--------------")
    
    hparams = make_hyperparam_string(USE_DATA_AUG, learning_rate_base, batch_size, kernel_size, dropout_rate,
                                     num_training, num_nodes_fc, activation_type)
    print('Saving the model to...')
    
    results_dir = os.path.join(params['results_base_directory'],hparams)
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(results_dir)

    print("--------------Done--------------")

    ############ Create a model ############
    print("--------------Create a model--------------")
    
    # Generate a model
    model = ResNet2D_classifier(data_num_rows=N_spatial[0], data_num_cols=N_spatial[1], num_classes=N_classes,
                                kernel_size=kernel_size, num_layers_each_block=num_layers_each_block,
                                num_chan_per_block=num_chan_per_block, activation_type=activation_type,
                                dropout_rate=dropout_rate, num_input_chans=N_bands, num_nodes_fc=num_nodes_fc)

    # Compile the model
    adam_opt = Adam(lr=learning_rate_base / batch_size, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['acc'])

    # Create a Tensorboard callback
    tbCallBack = TensorBoard(log_dir=results_dir, histogram_freq=0, write_graph=False, write_images=False)
    
    print("--------------Done--------------")

    ############ Train the model ############
    print("--------------Begin training the model--------------")

    # Possibly perform data augmentation
    from keras.preprocessing.image import ImageDataGenerator
    
    if USE_DATA_AUG:
        width_shift_range = 0.04
        height_shift_range = 0.04
        HORIZONTAL_FLIP = True
        VERTICAL_FLIP = True
        data_gen_args = dict(
            rotation_range=0.,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=HORIZONTAL_FLIP,
            vertical_flip=VERTICAL_FLIP,
            fill_mode = 'wrap')

        image_datagen = ImageDataGenerator(**data_gen_args)
    else:
        image_datagen = ImageDataGenerator()

    # Define a data generator to generate random batches
    def myGenerator(batch_size):
        for x_batch, y_batch in image_datagen.flow(x_training, labels_training, batch_size=batch_size, shuffle = True):
            yield (x_batch, y_batch)

    my_generator = myGenerator(batch_size)

    tic = timeit.default_timer()
    
    # Train the model
    hist = model.fit_generator(my_generator, steps_per_epoch=num_batch_per_epoch, epochs = num_epochs, initial_epoch = 0, verbose=2, callbacks = [tbCallBack])

    toc = timeit.default_timer()
    training_time = toc-tic
    print("Total training time = " + str(training_time))
    
    print("--------------Done--------------")

    print("--------------Make Predictions--------------")
    # In this example script, we use the training data as the test data
    x_test = x_training
    labels_test = labels_training
    num_test = num_training

    tic = timeit.default_timer()
    labels_predicted_test = model.predict(x_test)
    toc = timeit.default_timer()
    test_time = toc - tic
    print('Testing time (s) = ' + str(test_time) + '\n')

    ### Evaluation metrics

    # Classification accuracy
    labels_test_integer_format = np.argmax(labels_test, axis=1)
    labels_predicted_test_integer_format = np.argmax(labels_predicted_test, axis=1)

    acc_top2 = top_K_classification_accuracy(labels_predicted_test, labels_test_integer_format, K=2)
    acc_top1 = top_K_classification_accuracy(labels_predicted_test, labels_test_integer_format, K=1)

    # Confusion matrices
    confusion_matrix_results = confusion_matrix(labels_test_integer_format, labels_predicted_test_integer_format)
    print("Confusion matrix = ")
    print(confusion_matrix_results)

    # Precision, Recall, F1
    macro_avg = np.asarray(
        precision_recall_fscore_support(labels_test_integer_format, labels_predicted_test_integer_format,
                                        average='macro'))
    macro_avg_precision = macro_avg[0]
    macro_avg_recall = macro_avg[1]
    macro_avg_fscore = macro_avg[2]

    print('Top-1 accuracy (%) = ' + str(acc_top1) + '\n')
    print('Top-2 accuracy (%) = ' + str(acc_top2) + '\n')
    print('Macro-avg precision = ' + str(macro_avg_precision) + '\n')
    print('Macro-avg recall = ' + str(macro_avg_recall) + '\n')
    print('Macro-avg f-score = ' + str(macro_avg_fscore) + '\n')

    print("--------------Done--------------")

    print("--------------Compute Saliency Maps--------------")
    results_test_dir = os.path.join(results_dir, 'test')
    if not os.path.exists(results_test_dir):
        os.makedirs(results_test_dir)

    # Swap softmax with linear
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)

    for idx_rice in range(num_test):

        grads = visualize_saliency(model, layer_idx=-1, filter_indices=np.argmax(labels_test[idx_rice, :], axis=0),
                                   seed_input=x_test[idx_rice], backprop_modifier=None)

        ss_img = np.sqrt(np.sum(abs(x_test[idx_rice, :, :, :]) ** 2, axis=2))
        ss_img /= np.max(ss_img)

        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.imshow(ss_img, cmap='gray')
        plt.clim(0, 1)
        plt.axis('off')
        plt.colorbar()

        plt.subplot(3, 1, 2)
        plt.imshow((grads * np.uint8(255)).astype('uint8'), cmap='jet')
        plt.clim(0, 255)
        plt.axis('off')
        plt.colorbar()

        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * np.uint8(255))

        plt.subplot(3, 1, 3)
        ss_img = cv2.cvtColor((ss_img * np.uint8(255)).astype('uint8'), cv2.COLOR_GRAY2RGB)
        plt.imshow(overlay(jet_heatmap, ss_img, alpha=0.3))
        plt.clim(0, 255)
        plt.axis('off')
        plt.colorbar()

        plt.savefig(os.path.join(results_test_dir, str(idx_rice+1) + '.png'))
        plt.clf()

    print("--------------Done--------------")

    print("--------------Save the information--------------")

    # Write some information to files
    f = open(os.path.join(results_test_dir, 'testing_info.txt'), 'w')
    f.write("Rice types = " + str(rice_types) + "\n")
    f.write("Confusion matrix \n")
    f.write(str(confusion_matrix_results) + "\n")
    f.write("Normalization type = " + str(normalization_type) + "\n")
    f.write("# test samples = %d \n" % (num_test))
    f.write("Top-1 test accuracy = %f \n" % (acc_top1))
    f.write("Top-2 test accuracy = %f \n" % (acc_top2))
    f.write("Macro-avg precision = %f \n" % (macro_avg_precision))
    f.write("Macro-avg recall = %f \n" % (macro_avg_recall))
    f.write("Macro-avg f-score = %f \n" % (macro_avg_fscore))
    f.write("Test time (s) = " + str(test_time) + "\n")
    f.close()

    # Save confusion matrices
    plt.figure(1)
    plot_confusion_matrix(confusion_matrix_results, classes=rice_types, normalize=False, title='Confusion matrix')
    plt.savefig(os.path.join(results_test_dir,'confusionMatrix.png'))
    plt.clf()

    print("--------------Done--------------")

    print("--------------Save the information for the training phase--------------")
    
    import pandas as pd
    
    # Save the trained model
    model.save_weights(os.path.join(results_dir, 'trainedResNetB_weights.h5'))
    
    # Extract the training loss   
    training_loss = hist.history['loss']

    # Save the training loss
    df = pd.DataFrame(data={'training loss': training_loss},index=np.arange(num_epochs)+1)
    df.to_csv(os.path.join(results_dir,'training_loss.csv'))
    
    # Save the training loss as a figure
    plt.figure(1)
    plt.title('Loss')
    plt.plot(training_loss, color='b',label='Training')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir,'training_loss.png'))
    plt.clf()   
    
    # Write a file with general information
    f = open(os.path.join(results_dir,'training_info.txt'),'w')
    f.write(hparams + '\n')
    f.write('Rice types = ' + str(rice_types)+'\n')
    f.write('Training time (s) = %f \n' %(training_time))
    f.write('Normalization type = ' + str(normalization_type)+ '\n')
    f.write('# epochs = ' + str(num_epochs) + '\n')
    f.write('# training samples = %d \n' %(num_training))
    f.close()
    
    print("--------------Done--------------")
    

##################################################################################
###############################      Main Function     ###########################
##################################################################################

if __name__ == '__main__':
    
    # Parameters (mostly determined using validation datasets)
    params = dict()
    params['normalization_type'] = 'max'                    # Data normalization type
    params['rice_types'] = ['DM','JP','RB','LP','KN','ML']  # Rice types
    params['activation_type'] = 'swish'                     # Activation function
    params['batch_size'] = 4                                # Batch size
    params['kernel_size'] = 3                               # Kernel size
    params['dropout_rate'] = 0.0                            # Dropout rate
    params['num_nodes_fc'] = 512                            # Number of  nodes in the fully-connected layers
    params['num_layers_each_block'] = [8, 8, 12, 8]         # Number of layers per block
    params['num_chan_per_block'] = [128, 128, 256, 256]     # Number of filters in the conv layers

    # Additional parameters for training. In our experiment where we have the full training set, we set
    # USE_DATA_AUG = True and learning_rate_base = 0.005. However, in this example script, we change them to
    # USE_DATA_AUG = False and learning_rate_base = 0.00001 just to simplify our training process on this
    # example dataset
    params['USE_DATA_AUG'] = False  # Use data augmentation (In the paper, we set it to True)
    params['learning_rate_base'] = 0.00001  # Initial learning rate (In the paper, we set it to 0.05)
    params['num_epochs'] = 400  # Number of epochs
    params['results_base_directory'] = './results/'  # Directory of saving results

    # Add 'swish' activation
    if params['activation_type'] == 'swish':
        
        from keras.utils.generic_utils import get_custom_objects
        import keras.backend as K

        # Taken from https://github.com/dataplayer12/swish-activation/blob/master/MNIST/activations.ipynb
        def swish(x):
            beta = tf.Variable(initial_value=1.0,trainable=True)
            return x*tf.nn.sigmoid(beta*x)

        get_custom_objects().update({'swish': swish})

    createAndTrainResNetB(params)