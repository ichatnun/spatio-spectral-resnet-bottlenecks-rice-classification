# spatio-spectral-resnet-bottlenecks-rice-classification

Example data and Keras implementation of a deep convolutional neural network described in "Rice Classification Using Spatio-Spectral Deep Convolutional Neural Network" submitted to Computers and Electronics in Agriculture. This is slightly different from the version uploaded to arXiv. In particular, a deep residual network with bottleneck building blocks was used instead of DenseNet.

Overview
------
A non-destructive rice variety classification system that benefits from the synergy between hyperspectral imaging and deep convolutional neural network (CNN) is developed. The proposed method uses a hyperspectral imaging system to simultaneously acquire complementary spatial and spectral information of rice seeds. The rice variety of each rice seed is then determined from the acquired spatio-spectral data using a deep residual network with bottleneck building blocks (ResNet-B).

Files
------
* **script_run_proposed_RseNetB.py** is the main file. This script trains a residual network with bottleneck building blocks (ResNet-B) and then tests the trained model.

* **utils_rice.py** contains the modules needed for the main file.

* **x.npy** contains example datacubes of the processed rice dataset that can be used for training/testing. Each datacube is a three-dimensional 50x170x110 tensor: two spatial dimensions and one spectral dimension.

* **labels.npy** contains the corresponding labels of the datacubes stored in **x.npy**



