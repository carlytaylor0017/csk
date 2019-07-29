import argparse
import os

import cv2
import imutils
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from keras import backend as K
from keras import regularizers
from keras.activations import elu, relu
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten, Input,
                          MaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, Adam, Nadam, adam
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)

log_device_placement = True
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


img_width, img_height = 50, 50
nb_train_samples = 9691
nb_validation_samples = 9691
epochs = 500
batch_size = 25
nb_filters = 32
extra_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
act = 'elu'
opt = Adam(lr=0.0001, decay=1e-6)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

def make_model():
    """
    """
    if K.image_data_format() == 'channels_first':
        input_shape = (3 , img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, input_shape=input_shape))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(nb_filters, kernel_size))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(nb_filters + extra_filters, kernel_size))
    model.add(Activation(act))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(nb_filters + extra_filters))
    model.add(Activation(act))
    model.add(Dropout(0.1))

    model.add(Dense(nb_train_samples))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                   metrics=["accuracy", top_3_accuracy, top_5_accuracy])

    return model

def plot_acc(figure_path):
    """
    """
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(figure_path), dpi=600)
    plt.close()

def plot_losses(figure_path):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(figure_path), dpi=600)
    plt.close()

def generate_images(image_dir,model_dir,epochs):
    plot_acc(os.path.join(image_dir, 'loss_step_{}.png').format(epoch))
    plot_losses(os.path.join(image_dir, 'loss_step_{}.png').format(epoch))

def generate_checkpoints(model,image_dir, model_dir, epoch):
    """
    """
    #save model weights to disc
    model_checkpoint_base_name = os.path.join(model_dir, '{}_model_weights_step_{}.h5')
    model.save_weights(model_checkpoint_base_name.format('monochrome', epoch))


def train(train_data_dir, validation_data_dir, model_path, epochs):

    model = make_model()

    if model_path:
        model.load_weights(model_path, by_name=True)

    train_datagen = ImageDataGenerator(featurewise_std_normalization = True,
                                featurewise_center = True,
                                rotation_range=1,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.1,
                                zoom_range=0.1,
                                rescale=1. / 255,
                                horizontal_flip=False,
                                vertical_flip=False,
                                fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    generate_checkpoints(model,args.image_dir, args.model_dir, epochs)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1)

    for epoch in range(epochs):
        if epoch % 50 == 0:
            generate_images(args.image_dir, args.model_dir, epoch)

    if predict:
        generate_predictions(args.test_data_dir)

def generate_predictions(test_data_dir):

    test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

    nb_samples = len(filenames)

    test_generator.reset()
    predict = model.predict_generator(test_generator,steps = nb_samples)
    top_3_index = np.fliplr(np.argsort(predict, axis=1)[:, -3:]).T
    top_3_percent = np.fliplr(np.sort(predict, axis=1)[:, -3:]).T

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    arr_1 = top_3_index[:1][0]
    arr_2 = top_3_index[1:2][0]
    arr_3 = top_3_index[2:3][0]

    prediction_1 = [labels[k] for k in arr_1]
    prediction_2 = [labels[k] for k in arr_2]
    prediction_3 = [labels[k] for k in arr_3]

    percent_1 = top_3_percent[0]
    percent_2 = top_3_percent[1]
    percent_3 = top_3_percent[2]

    filenames=test_generator.filenames

    results=pd.DataFrame({"Filename":filenames,
                          "Prediction_1":prediction_1,
                          "Percent_1":percent_1,
                          "Prediction_2":prediction_2,
                          "Percent_2":percent_2,
                          "Prediction_3":prediction_3,
                          "Percent_3":percent_3})

    return results

def main(train_data_dir, validation_data_dir, model_path, epochs):
    train(train_data_dir, validation_data_dir, model_path, epochs)

if __name__ == '__main__':
    # Note: if pre-trained models are passed in we don't take the steps they've been trained for into account
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_data_dir", "-t", required=True,
                        help="Directory of train images")
    parser.add_argument("--validation_data_dir", "-v", required=True,
                            help="Directory of validation images")
    parser.add_argument("--test_data_dir", "-ts", required=False,
                            help="Directory of test images")
    parser.add_argument("--image_dir", "-i", required=True,
                        help="Directory to output image files to")
    parser.add_argument("--model_dir", "-m", required=True,
                        help="Directory to output model files to")
    parser.add_argument("--model_path", "-p", required=False,
                        help="Directory for model weights")
    args = parser.parse_args()

    main(args.train_data_dir, args.validation_data_dir, args.model_path, epochs)
