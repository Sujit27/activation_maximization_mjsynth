# Activation maximization with mjsynth dataset
## Activation maximization
Activation maximization is optimization in the input space to maximize a particular neuron in the last layer without changing the weights of the neural network. For example, following are the images generated from a model trained on mnist data by picking and maximizing one of the 10 neurons in the last layer

![Five](https://github.com/Sujit27/activation_maximization_mjsynth/blob/master/deep_dream_mnist/mnist_deepdreamt/five.png)


![Eight](https://github.com/Sujit27/activation_maximization_mjsynth/blob/master/deep_dream_mnist/mnist_deepdreamt/eight.png)

## Mjsynth
[Mjsynth](https://www.robots.ox.ac.uk/~vgg/data/text/) is synthetic word dataset for text recognition with around 9 million images from approximately 80,000 words in english

Prerequisites

```pip install torch```

```pip install --user --upgrade git+https://github.com/anguelos/dagtasets```

Download dataset

```python3 /subset_dataset/create_dataset_files/create_dataset.py $download_dir```

where download_dir is the location where you want to download the dataset. Make sure minimum of 50gb is available

Sample activation maximization with different gaussian filters and parameters

![Changing learning rate](https://github.com/Sujit27/activation_maximization_mjsynth/blob/master/dreams/diff_lr/count_skype_admin_grind_0.001_0.01_0.05_0.1_0.5_learningRate.png)


![Changing sigma of gaussian filter](https://github.com/Sujit27/activation_maximization_mjsynth/blob/master/dreams/diff_gaussian_filter/count_skype_admin_grind_0.3_0.4_0.5_0.6_0.7_gaussianFilterSigma.png)
