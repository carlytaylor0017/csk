
"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028
The improved WGAN has a term in the loss function which penalizes the network if its
gradient norm moves away from 1. This is included because the Earth Mover (EM) distance
used in WGANs is only easy to calculate for 1-Lipschitz functions (i.e. functions where
the gradient norm has a constant upper bound of 1).
The original WGAN paper enforced this by clipping weights to very small values
[-0.01, 0.01]. However, this drastically reduced network capacity. Penalizing the
gradient norm is more natural, but this requires second-order gradients. These are not
supported for some tensorflow ops (particularly MaxPool and AveragePool) in the current
release (1.0.x), but they are supported in the current nightly builds
(1.1.0-rc1 and higher).
To avoid this, this model uses strided convolutions instead of Average/Maxpooling for
downsampling. If you wish to use pooling operations in your discriminator, please ensure
you update Tensorflow to 1.1.0-rc1 or higher. I haven't tested this with Theano at all.
The model saves images using pillow. If you don't have pillow, either install it or
remove the calls to generate_images.
"""
import argparse
import os
import pickle
import sys
from functools import partial

import matplotlib
import numpy as np
from keras import applications
from keras import backend as K
from keras import layers, models, optimizers
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Convolution2D
from keras.layers.merge import _Merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

matplotlib.rcParams.update({'axes.titlesize': 5})

log_device_placement = True
batch_size = 11
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
img_height = 28
img_width = 28
img_channels = 1
epochs = 10000


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
    and outputs images of size 28x28x1."""
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        bn_axis = 1
    else:
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
        bn_axis = -1
    model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure
    # its output also lies in this range. The sigmoid function will force the
    #images to be black and white.
    model.add(Convolution2D(img_channels, (5, 5), padding='same', activation='sigmoid'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Convolution2D(64, (5, 5), padding='same',
                  input_shape=(img_channels, img_height, img_width)))
    else:
        model.add(Convolution2D(64, (5, 5), padding='same',
                  input_shape=(img_height, img_width, img_channels)))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal',
                            strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same',
                            strides=[2, 2]))

    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(img_channels, kernel_initializer='he_normal'))
    return model


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def plot_batch(image_batch, figure_path, label_batch=None, vmin=0, vmax=255, scale=True):
    """Plots a batch of images and their corresponding label(s)/annotations, saving the plot to disc.
    :param image_batch: Batch of images to be plotted.
    :param figure_path: Full path of the filename the plot will be saved as.
    :param label_batch: Batch of labels corresponding to `image_batch`.
       Labels will be displayed along w/ their corresponding image.
    """

    batch_size = len(image_batch)
    assert batch_size >= 1
    assert isinstance(image_batch, np.ndarray), 'image_batch must be an np array.'
    # for gray scale images if image_batch.shape == (img_height, img_width, 1) plt requires this to be reshaped
    if image_batch.shape[-1] == 1:
        image_batch = np.reshape(image_batch, newshape=image_batch.shape[:-1])
    # plot images in rows and columns
    # `+ 2` prevents plt.subplots from throwing: `TypeError: 'AxesSubplot' object does not support indexing` when batch_size < 10
    if batch_size < 10:
        nb_rows = 1
        nb_columns = 2
    else:
        nb_rows = batch_size // 10 + 1
        nb_columns = batch_size // 10 + 2
    _, ax = plt.subplots(nb_rows, nb_columns, sharex=True, sharey=True)
    for i in range(nb_rows):
        for j in range(nb_columns):
            try:
                x = image_batch[i * nb_columns + j]
                if scale:
                    x = x + max(-np.min(x), 0)
                    x_max = np.max(x)
                    if x_max != 0:
                        x /= x_max
                    x *= 255

                ax[i][j].imshow(x.astype('uint8'), vmin=vmin, vmax=vmax,
                                interpolation='lanczos', cmap='gray')
                if label_batch is not None:
                    ax[i][j].set_title(label_batch[i * nb_columns + j])
                ax[i][j].set_axis_off()
            except IndexError:
                break

    plt.savefig(os.path.join(figure_path), dpi=600)
    plt.close()


def adversarial_training(data_dir, generator_model_path, discriminator_model_path):
        """trains the generator and discrminator and saves weights and images every
        50 epochs
        """

    data_generator = image.ImageDataGenerator(data_format='channels_last',
                                              rescale=1. / 255)

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale',
                                  'class_mode': None,
                                  'batch_size': batch_size}

    real_image_generator = data_generator.flow_from_directory(directory=data_dir,
                                                              **flow_from_directory_params)

    def get_image_batch():

        img_batch = real_image_generator.next()
        # keras generators may generate an incomplete batch for the last batch in an epoch of data
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        return img_batch

    def generate_images(generator_model, output_dir, model_dir, epoch):
        """Feeds random seeds into the generator and tiles and saves the output to a PNG
        file."""
        g_z = generator_model.predict(np.random.rand(10, 100))
        if epoch % 50 == 0:
            # save a batch of generated and real images to disc
            plot_batch(g_z, os.path.join(output_dir, 'batch_image_step_{}.png').format(epoch))
            # save model weights to disc
            model_checkpoint_base_name = os.path.join(model_dir, '{}_model_weights_step_{}.h5')
            generator_model.save_weights(model_checkpoint_base_name.format('generator', epoch))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', epoch))

    # Now we initialize the generator and discriminator.
    generator = make_generator()
    discriminator = make_discriminator()

    # The generator_model is used when we want to train the generator layers.
    # As such, we ensure that the discriminator layers are not trainable.
    # Note that once we compile this model, updating .trainable will have no effect within
    # it. As such, it won't cause problems if we later set discriminator.trainable = True
    # for the discriminator_model, as long as we compile the generator_model first.
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input = Input(shape=(100,))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input],
                            outputs=[discriminator_layers_for_generator])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=wasserstein_loss)

    # Now that the generator_model is compiled, we can make the discriminator
    # layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random
    # noise seeds as input. The noise seed is run through the generator model to get
    # generated images. Both real and generated images are then run through the
    # discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    x = get_image_batch()
    real_samples = Input(shape=(img_height, img_width, img_channels))
    generator_input_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples,
    # to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()([real_samples,
                                                generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never
    # really use the discriminator output for these samples - we're only running them to
    # get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get
    # gradients. However, Keras loss functions can only have two arguments, y_true and
    # y_pred. We get around this by making a partial() of the function with the averaged
    # samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'

    # Keras requires that inputs and outputs have the same number of samples. This is why
    # we didn't concatenate the real samples and generated samples before passing them to
    # the discriminator: If we had, it would create an output with 2 * BATCH_SIZE samples,
    # while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.

    # If we don't concatenate the real and generated samples, however, we get three
    # outputs: One of the generated samples, one of the real samples, and one of the
    # averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples,
                                        generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
    # the real and generated samples, and the gradient penalty loss for the averaged samples
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    # We make three label vectors for training. positive_y is the label vector for real
    # samples, with value 1. negative_y is the label vector for generated samples, with
    # value -1. The dummy_y vector is passed to the gradient_penalty loss function and
    # is not used.
    positive_y = np.ones((batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    if generator_model_path:
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        discriminator_loss = []
        generator_loss = []
        # train the discriminator
        for i in range(1):
            for j in range(TRAINING_RATIO):
                noise = np.random.rand(batch_size, 100).astype(np.float32)
                discriminator_loss.append(discriminator_model.train_on_batch(
                    [x, noise],
                    [positive_y, negative_y, dummy_y]))
            generator_loss.append(generator_model.train_on_batch(np.random.rand(batch_size,
                                                                                100),
                                                                 positive_y))
        generate_images(generator, args.output_dir, args.model_dir, epoch)

def main(data_dir, generator_model_path, discriminator_model_path):
    adversarial_training(data_dir, generator_model_path, discriminator_model_path)

if __name__ == '__main__':
    # Note: if pre-trained models are passed in we don't take the steps they've been trained for into account
    parser = argparse.ArgumentParser(description="Improved Wasserstein GAN "
                                                 "implementation for Keras.")
    parser.add_argument("--data_dir", "-i", required=True,
                        help="Directory of images")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to output image files to")
    parser.add_argument("--model_dir", "-m", required=True,
                        help="Directory to output model files to")
    parser.add_argument("--generator_model_path", "-g", required=False,
                        help="Directory for generated model weights")
    parser.add_argument("--discriminator_model_path", "-d", required=False,
                        help="Directory for discriminator model weights")
    args = parser.parse_args()

    main(args.data_dir, args.generator_model_path, args.discriminator_model_path)
