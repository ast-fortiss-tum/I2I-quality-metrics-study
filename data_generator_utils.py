# data_generator_utils.py




import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import os
import cv2
import matplotlib.pyplot as plt
import re
from collections import OrderedDict
from transformers import TFSegformerForSemanticSegmentation
import os
import json


OUTPUT_CHANNELS = 3
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512


def natural_sort_key(s):
    """
    The `natural_sort_key` function returns a key that can be used for natural sorting of strings, where
    numbers are sorted numerically and letters are sorted alphabetically.
    
    :param s: The parameter `s` is a string that represents the input value that we want to generate a
    natural sort key for
    :return: The function `natural_sort_key` returns a list comprehension that converts each element in
    the input string `s` into either an integer (if the element is a digit) or a lowercase string (if
    the element is not a digit).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_h5_to_dictionary(file_path):
    """
    The function `load_h5_to_dictionary` loads data from an HDF5 file into a dictionary, where each key
    in the dictionary corresponds to a group in the HDF5 file and the corresponding value is a list of
    arrays containing the datasets within that group.
    
    :param file_path: The file path is the path to the H5 file that you want to load into a dictionary
    :return: a dictionary containing the data loaded from the h5 file.
    """
    data_dict = OrderedDict()
    with h5py.File(file_path, 'r') as hf:
        for key in hf.keys():
            array_list = []
            group = hf[key]
            sorted_dataset_names = sorted(group.keys(), key=natural_sort_key)
            for dataset_name in sorted_dataset_names:
                array_list.append(np.array(group[dataset_name]))
            data_dict[key] = array_list
    return data_dict

def find_lowest_dimensions(images_list1, images_list2):
    """
    The function `find_lowest_dimensions` takes in two lists of images and returns the lowest height and
    width among the first images in each list.
    
    :param images_list1: A list of images represented as numpy arrays. Each image has a shape (height,
    width, channels)
    :param images_list2: The `images_list2` parameter is a list of images
    :return: the lowest height and width dimensions among the first images in the two input lists.
    """
    lowest_height = min(images_list1[0].shape[0], images_list2[0].shape[0])
    lowest_width = min(images_list1[0].shape[1], images_list2[0].shape[1])
    return lowest_height, lowest_width

def crop_images_to_lowest_dimensions(images_list, lowest_height, lowest_width):
    """
    The function crops a list of images to the lowest specified height and width.
    
    :param images_list: A list of images that you want to crop
    :param lowest_height: The lowest height is the desired height that you want to crop the images to
    :param lowest_width: The lowest width is the desired width that you want to crop the images to
    :return: a list of cropped images, where each image has been cropped to the lowest specified height
    and width.
    """
    cropped_images = [image[:lowest_height, :lowest_width,:] for image in images_list]
    return cropped_images

def crop_1d_to_lowest_dimensions(images_list, lowest_height, lowest_width):
    """
    The function `crop_1d_to_lowest_dimensions` takes a list of images and crops each image to the
    specified lowest height and width.
    
    :param images_list: A list of 1-dimensional images. Each image is represented as a numpy array
    :param lowest_height: The lowest height is the desired height that you want to crop the images to
    :param lowest_width: The lowest width is the desired width that you want to crop the images to
    :return: a list of cropped images, where each image has been cropped to the lowest height and width
    specified.
    """
    cropped_images = [image[:lowest_height, :lowest_width] for image in images_list]
    return cropped_images

def display(display_list):
    """
    The function `display` takes a list of images and displays them in a grid with corresponding titles.
    
    :param display_list: The `display_list` parameter is a list that contains four elements. Each
    element represents an image or an array that you want to display. The four elements in the list are:
    """
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask","diff"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        print(display_list[i].shape)
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()

def map_to_desired_structure(input_image,output_image):
    """
    The function "map_to_desired_structure" takes an input image and an output image and returns a
    dictionary with the input image as the value for the key "pixel_values" and the output image as the
    value for the key "labels".
    
    :param input_image: The input_image parameter is the image data that you want to map to the desired
    structure. It could be a 2D or 3D array representing the pixel values of the image
    :param output_image: The output_image parameter is the desired structure or format that you want the
    input_image to be mapped to. It could be a different image format, such as converting from one file
    type to another, or it could be a different data structure, such as converting from a list to a
    dictionary. The specific
    :return: A dictionary is being returned with two keys: 'pixel_values' and 'labels'. The value
    associated with the 'pixel_values' key is the input_image, and the value associated with the
    'labels' key is the output_image.
    """
    return {'pixel_values': input_image, 'labels': output_image}

def map_values_kitti(value):
    """
    The function `map_values_kitti` maps specific input values to corresponding output values based on
    predefined conditions.
    
    :param value: The parameter "value" is a variable that represents a numerical value
    :return: The function `map_values_kitti` returns an integer value based on the input value. The
    possible return values are 0, 1, 2, 3, or 4.
    """
    if value in [1, 2, 6, 7, 8, 16, 21, 255]:
        return 1
    elif value in [11, 12]:
        return 2
    elif value in [13, 14, 15, 17]:
        return 3
    elif value == 10:
        return 4
    elif value == 0:
        return 0
    else:
        return 1 
    
def map_values_donkey(value):
    return int(value) 
    
def create_mask(pred_mask):
    """
    The function `create_mask` takes a predicted mask as input, performs some operations on it, and
    returns the resulting mask.
    
    :param pred_mask: The parameter `pred_mask` is a tensor representing the predicted mask. It is
    expected to have shape `(batch_size, num_classes, height, width)`
    :return: the predicted mask.
    """
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]
    
def cast_to_int32(image):
    """
    The function takes an image as input, normalizes its pixel values to the range [0, 1], and then
    converts it to uint8 by scaling the values back to the range [0, 255].
    
    :param image: The "image" parameter is a tensor representing an image
    :return: the image after converting its pixel values to int32.
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.cast(image * 255, tf.int32)
    return image

def random_crop(image):
    """
    The function `random_crop` takes an image as input and returns a randomly cropped version of the
    image with the specified height and width.

    :param image: The input image that you want to randomly crop
    :return: a randomly cropped image with dimensions [IMG_HEIGHT, IMG_WIDTH, 3].
    """
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

def normalize(image):
    """
    The `normalize` function takes an image as input, converts it to float32 data type, and normalizes
    the pixel values to the range [-1, 1].
    
    :param image: The "image" parameter is a tensor representing an image
    :return: The normalized image is being returned.
    """
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):
    """
    The function randomly applies various transformations to an image, including resizing, cropping, and
    flipping.
    
    :param image: The "image" parameter is the input image that you want to apply random jitter to
    :return: the image after applying random jitter operations such as resizing, random cropping, and
    random flip left-right.
    """
    image = tf.image.resize(image, [520, 520],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image_train(input):
    """
    The function preprocesses an input image for training by applying random jitter, transposing the
    image, and normalizing it.
    
    :param input: The input parameter is a dictionary that contains the pixel values of an image
    :return: the preprocessed image.
    """
    image=input["pixel_values"]
    image = random_jitter(tf.transpose(image, (1, 2, 0)))
    image = normalize(image)
    return image

def preprocess_image_test(input):
    """
    The function preprocesses an input image by applying random jitter, transposing the image, and
    normalizing it.
    
    :param input: The input parameter is a dictionary that contains the pixel values of an image
    :return: the preprocessed image.
    """
    image=input["pixel_values"]
    image = random_jitter(tf.transpose(image, (1, 2, 0)))
    image = normalize(image)
    return image

def generate_images_cyclegan(model, test_input,plot,height,width):
    """
    The function `generate_images_cyclegan` takes a model, test input, and other parameters, and
    generates and displays predicted images based on the input using the CycleGAN model.
    
    :param model: The model is the CycleGAN model that has been trained to generate images. It takes an
    input image and generates a corresponding output image
    :param test_input: The test_input parameter is a dictionary that contains the input image data.
    Specifically, it contains the pixel values of the input image
    :param plot: The "plot" parameter is a boolean value that determines whether or not to display the
    generated images using matplotlib.pyplot. If set to True, the function will display the input image
    and the predicted image side by side. If set to False, the function will not display the images but
    will still return the
    :param height: The height parameter is the desired height of the output image
    :param width: The width parameter is the desired width of the output image
    :return: the predicted image generated by the model.
    """
    image=test_input["pixel_values"]
    image = tf.transpose(image[0], (1, 2, 0))
    image = tf.image.resize(image, [512, 512],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize(image)
    image2=tf.expand_dims(image, 0)
    prediction = model(image2)
    if plot:
        plt.figure(figsize=(12, 12))

    display_list = [image, prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        if plot:
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
        image=display_list[i]
        image = tf.image.resize(image, [height,width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image=image * 0.5 + 0.5
        if plot:
            plt.imshow(image)
            plt.axis('off')
    if plot:
        plt.show()
    return prediction[0]





def downsample(filters, size, apply_batchnorm=True):
    """
    The function `downsample` takes in parameters for filters, size, and apply_batchnorm, and returns a
    sequential model that performs downsampling on an input image.
    
    :param filters: The "filters" parameter specifies the number of filters (or channels) in the
    convolutional layer. It determines the depth or dimensionality of the output feature maps
    :param size: The "size" parameter refers to the size of the filters used in the convolutional layer.
    It determines the spatial extent of the filters. For example, if size=3, it means that the filters
    will have a size of 3x3
    :param apply_batchnorm: The `apply_batchnorm` parameter is a boolean value that determines whether
    batch normalization should be applied after the convolutional layer. If `apply_batchnorm` is set to
    `True`, batch normalization will be applied. If it is set to `False`, batch normalization will not
    be applied, defaults to True (optional)
    :return: a sequential model that consists of a convolutional layer followed by batch normalization
    and a leaky ReLU activation function.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def Generator():
    """
    The function `Generator` creates a generator model for image synthesis using a U-Net architecture.
    :return: The code is returning a TensorFlow Keras model that takes an input tensor of shape [512,
    512, 3] and outputs a tensor of shape [256, 256, 3].
    """
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    """
    The Discriminator function creates a model that takes in two input images and outputs a
    single-channel image.
    :return: The code is returning a TensorFlow Keras model that takes two input images (input_image and
    target_image) and outputs a 30x30x1 tensor. This model is a discriminator, which is typically used
    in adversarial networks to distinguish between real and fake images.
    """ 
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def upsample(filters, size, apply_dropout=False):
    """
    The function `upsample` creates a sequential model in TensorFlow for upsampling an image using
    transpose convolution, batch normalization, and optional dropout.
    
    :param filters: The "filters" parameter specifies the number of filters (or channels) in the output
    of the convolutional transpose layer. It determines the depth of the output feature map
    :param size: The size parameter specifies the size of the filters in the Conv2DTranspose layer. It
    determines the spatial extent of the filters and affects the output size of the layer
    :param apply_dropout: The parameter "apply_dropout" is a boolean value that determines whether or
    not to apply dropout regularization. If set to True, dropout regularization will be applied with a
    dropout rate of 0.5. If set to False, dropout regularization will not be applied, defaults to False
    (optional)
    :return: a sequential model that consists of a transposed convolutional layer, batch normalization
    layer, optional dropout layer, and a ReLU activation function.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
    
    
def generate_images_pix2pix(model, test_input,plot):
    """
    The function generates and displays predicted images using a Pix2Pix model.
    
    :param model: The model is the Pix2Pix model that has been trained to generate images. It takes an
    input image and generates a corresponding output image
    :param test_input: The test_input parameter is the input image that you want to generate a
    prediction for using the pix2pix model. It should be a tensor representing the image, typically of
    shape (1, height, width, channels), where height and width are the dimensions of the image and
    channels is the number of
    :param plot: The "plot" parameter is a boolean value that determines whether or not to display the
    generated images using matplotlib.pyplot. If set to True, the function will display the input image
    and the predicted image side by side. If set to False, the function will not display the images but
    will still return the
    :return: the predicted image.
    """
    prediction = model(test_input, training=True)
    

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    if plot:
        plt.figure(figsize=(15, 15))
        for i in range(2):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(tf.cast(display_list[i]* 0.5 + 0.5, tf.float32))
            plt.axis('off')
        plt.show()
    return prediction[0]


def map_values_car(value):
    return int(value)

def auto_canny(image, sigma=0.66):
    """
    The function `auto_canny` applies automatic Canny edge detection to an image using the computed
    median of the pixel intensities.

    :param image: The input image on which the Canny edge detection will be applied
    :param sigma: The sigma parameter in the auto_canny function is used to control the threshold values
    for the Canny edge detection algorithm. It determines the range of pixel intensities that will be
    considered as edges
    :return: the edged image after applying automatic Canny edge detection.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def map_to_r(number):
    """
    The function `map_to_r` takes a number as input and returns the corresponding value from a
    predefined list of red color values.
    
    :param number: The parameter "number" represents an integer value that is used to index into the
    "colorr" list
    :return: The function `map_to_r` returns the value at the index `number` in the `colorr` list if
    `number` is less than 20. Otherwise, it returns 0.
    """
    colorr = [255, 128, 0, 0, 128, 0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 255, 128, 255, 0, 128]
    if number<20:
      return int(colorr[number])
    else:
      return 0

def map_to_g(number):
    """
    The function `map_to_g` takes a number as input and returns the corresponding value from the
    `colorg` list if the number is less than 20, otherwise it returns 0.
    
    :param number: The parameter "number" represents the index of the colorg list
    :return: The function `map_to_g` returns the value at the index `number` in the `colorg` list if
    `number` is less than 20. Otherwise, it returns 0.
    """

    colorg = [0, 128, 255, 128, 0, 0, 128, 0, 0, 128, 0, 128, 0, 128, 255, 128, 0, 128, 255, 128]

    if number<20:
      return int(colorg[number])
    else:
      return 0

def map_to_b(number):
    """
    The function `map_to_b` takes a number as input and returns the corresponding value from a
    predefined list of blue color values, or 255 if the number is greater than or equal to 20.
    
    :param number: The parameter "number" represents an integer value that is used to index into the
    "colorb" list
    :return: If the input number is less than 20, the function will return the corresponding value from
    the colorb list. Otherwise, it will return 255.
    """

    colorb = [0, 0, 0, 128, 128, 255, 255, 128, 0, 128, 128, 255, 128, 0, 0, 0, 128, 255, 255, 128]

    if number<20:
      return int(colorb[number])
    else:
      return 255

def rgb2gray(rgb):
    """
    The function `rgb2gray` converts an RGB image to grayscale using the formula `gray = 0.2989 * red +
    0.5870 * green + 0.1140 * blue`.
    
    :param rgb: The parameter `rgb` is expected to be a 3-dimensional numpy array representing an image
    in RGB format. The dimensions of the array should be (height, width, 3), where the last dimension
    represents the red, green, and blue channels of the image
    :return: The function `rgb2gray` returns a grayscale version of the input RGB image.
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_gan_indexes(dataset_index_list_test,loaded_dictionary_images_real,loaded_dictionary_images_sim,pattern):
    """
    The function `get_gan_indexes` takes in a list of dataset indexes, dictionaries of loaded images,
    and a pattern, and returns two dictionaries containing the train and test indexes for each dataset
    based on the pattern.
    
    :param dataset_index_list_test: A list of dataset indexes that you want to generate GAN train and
    test indexes for
    :param loaded_dictionary_images_real: The parameter "loaded_dictionary_images_real" is a dictionary
    that contains loaded images for each dataset index. The keys of the dictionary are dataset indexes,
    and the values are lists of images for that dataset index
    :param loaded_dictionary_images_sim: The parameter "loaded_dictionary_images_sim" is a dictionary
    that contains loaded images for each dataset index. The keys of the dictionary are dataset indexes,
    and the values are lists of images for that dataset index
    :param pattern: The "pattern" parameter is a string that determines the distribution of train and
    test indexes for each dataset. It is used to alternate between assigning indexes to the train and
    test sets. The pattern should consist of 't' and 's' characters, where 't' represents train indexes
    and 's
    :return: two dictionaries: `train_indexes_gan` and `test_indexes_gan`.
    """
    train_indexes_gan = {}
    test_indexes_gan = {}
    pattern_pointer = 0
    for dataset_index in dataset_index_list_test:
      train_dataset_indices_inner = []
      test_dataset_indices_inner = []
    
      for index in range(0, min(len(loaded_dictionary_images_real[dataset_index]),len(loaded_dictionary_images_sim[dataset_index]))):
        pattern_char = pattern[pattern_pointer]
        if pattern_char == 't':
            train_dataset_indices_inner.append(index)
        else:
            test_dataset_indices_inner.append(index)
        pattern_pointer = (pattern_pointer + 1) % len(pattern)
      train_indexes_gan[dataset_index]=train_dataset_indices_inner
      test_indexes_gan[dataset_index]=test_dataset_indices_inner
    
    print("GAN train and test")
    for dataset_index in dataset_index_list_test:
        print("Dataset", dataset_index)
        print("Train: ",len(train_indexes_gan[dataset_index]))
        print("Test: ",len(test_indexes_gan[dataset_index]))
    return train_indexes_gan,test_indexes_gan

def load_segmentation_model(checkpoint_file_path,task_type):
    """
    The function `load_segmentation_model` loads a semantic segmentation model for either the "kitti" or
    "donkey" task type, with a specified checkpoint file path.
    
    :param checkpoint_file_path: The `checkpoint_file_path` parameter is the file path to the saved
    weights of the segmentation model. It should be a string representing the file path where the
    weights are stored
    :param task_type: The `task_type` parameter is used to specify the type of segmentation task you are
    working on. It can take two possible values:
    :return: a segmentation model.
    """
    model_checkpoint = "nvidia/mit-b0"
    if task_type=="kitti":
        id2label = {0: 'Road', 1: 'Environment', 2: 'Person', 3: 'Vehicle', 4: 'Sky'}
    elif task_type=="donkey":
        id2label = {0: 'Other', 1: 'Road', 2: 'LaneMark'}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)
    print(len(id2label))
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.load_weights(checkpoint_file_path)
    return model

def crop_data_dictionaries(task_type,dataset_index_list_test,loaded_dictionary_images_real,loaded_dictionary_images_sim,loaded_semantic_id_real,loaded_semantic_id_sim):
    """
    The function `crop_data_dictionaries` crops the images and semantic IDs in the loaded data
    dictionaries to the lowest dimensions found in the specified dataset indices.
    
    :param task_type: The task type is a string that specifies the type of task being performed. It
    could be "kitti" or any other task type
    :param dataset_index_list_test: dataset_index_list_test is a list of indices that specify which
    datasets to process. These indices correspond to specific datasets within the loaded dictionaries
    and semantic IDs
    :param loaded_dictionary_images_real: The parameter "loaded_dictionary_images_real" is a dictionary
    that contains loaded images for a specific dataset. Each key in the dictionary represents a dataset
    index, and the corresponding value is the loaded image for that dataset index
    :param loaded_dictionary_images_sim: The parameter "loaded_dictionary_images_sim" is a dictionary
    that contains loaded images for simulation data. Each key in the dictionary represents a dataset
    index, and the corresponding value is the loaded image for that dataset index
    :param loaded_semantic_id_real: The parameter `loaded_semantic_id_real` is a dictionary containing
    semantic IDs for real images in a dataset. Each key in the dictionary represents a dataset index,
    and the corresponding value is a list of semantic IDs for the images in that dataset
    :param loaded_semantic_id_sim: The parameter "loaded_semantic_id_sim" is a dictionary that contains
    semantic IDs for simulated images. It is used in the function "crop_data_dictionaries" to crop the
    semantic IDs to the lowest dimensions found in the images
    :return: the updated dictionaries `loaded_dictionary_images_real`, `loaded_dictionary_images_sim`,
    `loaded_semantic_id_real`, and `loaded_semantic_id_sim`.
    """
    lowest_height=10000
    lowest_width=10000
    for dataset_index in dataset_index_list_test:
      height_check, width_check = find_lowest_dimensions(loaded_dictionary_images_real[dataset_index], loaded_dictionary_images_sim[dataset_index])
      lowest_height=min(height_check,lowest_height)
      lowest_width=min(width_check,lowest_width)
    print(lowest_height,lowest_width)
    
    for dataset_index in dataset_index_list_test:
    
        loaded_dictionary_images_real[dataset_index] = crop_images_to_lowest_dimensions(loaded_dictionary_images_real[dataset_index], lowest_height, lowest_width)
        loaded_dictionary_images_sim[dataset_index] = crop_images_to_lowest_dimensions(loaded_dictionary_images_sim[dataset_index], lowest_height, lowest_width)
    
        if task_type=="kitti":
            loaded_semantic_id_real[dataset_index] = crop_1d_to_lowest_dimensions(loaded_semantic_id_real[dataset_index], lowest_height, lowest_width)
            loaded_semantic_id_sim[dataset_index] = crop_1d_to_lowest_dimensions(loaded_semantic_id_sim[dataset_index], lowest_height, lowest_width)
    return loaded_dictionary_images_real,loaded_dictionary_images_sim,loaded_semantic_id_real,loaded_semantic_id_sim




def plot_dataset_pairs(task_type,dataset_index_list_test,loaded_dictionary_images_real,loaded_dictionary_images_sim,loaded_semantic_id_real,loaded_semantic_id_sim,images_number,loaded_bounding_real,road):
    """
    The function `plot_dataset_pairs` takes in various parameters related to a dataset and task type,
    and plots pairs of images and masks for visualization purposes.
    
    :param task_type: The type of task being performed. It can be either "kitti" or "donkey"
    :param dataset_index_list_test: A list of dataset indices for which the plots will be generated
    :param loaded_dictionary_images_real: The parameter `loaded_dictionary_images_real` is a dictionary
    that contains the real images for each dataset. The keys of the dictionary are dataset indices, and
    the values are lists of images
    :param loaded_dictionary_images_sim: The parameter `loaded_dictionary_images_sim` is a dictionary
    that contains the simulated images for each dataset index. The keys of the dictionary are the
    dataset indices, and the values are lists of simulated images
    :param loaded_semantic_id_real: The parameter `loaded_semantic_id_real` is a dictionary that
    contains the semantic segmentation masks for the real images in the dataset. The keys of the
    dictionary represent the dataset index, and the values are arrays of semantic segmentation masks for
    each image in the dataset
    :param loaded_semantic_id_sim: The parameter `loaded_semantic_id_sim` is a dictionary that contains
    the semantic segmentation masks for the simulated images. The keys of the dictionary represent the
    dataset index, and the values are lists of semantic segmentation masks for each image in the dataset
    :param images_number: The parameter `images_number` represents the number of images to be plotted in
    the dataset pairs
    :param loaded_bounding_real: The parameter `loaded_bounding_real` is a dictionary that contains the
    bounding box information for the real images in the dataset. The keys of the dictionary are dataset
    indices, and the values are lists of bounding box information for each image in the dataset. Each
    bounding box information is represented as a tuple `(
    :param road: The parameter "road" is used to specify the ID of the road class in the semantic
    segmentation masks. It is used to create binary masks for the road class in both the real and
    simulated images. These masks are then used to calculate the difference mask, which shows the areas
    where the road class differs
    """
    if task_type=="kitti":
        additional_id_init=13
        boxes_real_dict={}
        for dataset_index in dataset_index_list_test:
            boxes_real={}
            for image_index,label,x_min, y_min, x_max, y_max in loaded_bounding_real[dataset_index]:
               if image_index not in boxes_real:
                   boxes_real[image_index] = [[label,x_min, y_min, x_max, y_max]]
               else:
                   boxes_real[image_index].append([label,x_min, y_min, x_max, y_max])
            
            
            boxes_real_dict[dataset_index]=boxes_real
       
    for dataset_index in dataset_index_list_test:
      print(dataset_index)
      print(dataset_index)
      for i in range(1, images_number+1):
          if task_type=="kitti":
            fig, axs = plt.subplots(3, 3, figsize=(15, 6))
          elif task_type=="donkey":
              fig, axs = plt.subplots(2, 3, figsize=(15, 6))
          plt.subplots_adjust(wspace=0.2, hspace=0.4)
    
          # Plot the real and fake images side by side in the first row
          # if task_type=="kitti":
          axs[0,0].imshow(cv2.cvtColor(loaded_dictionary_images_real[dataset_index][i], cv2.COLOR_BGR2RGB))
          axs[0,0].set_title('Real Image')
          axs[0,1].imshow(cv2.cvtColor(loaded_dictionary_images_sim[dataset_index][i], cv2.COLOR_BGR2RGB))
          axs[0,1].set_title('Sim Image')
          # elif task_type=="donkey":
          #     axs[0].imshow(cv2.cvtColor(loaded_dictionary_images_real[dataset_index][i], cv2.COLOR_BGR2RGB))
          #     axs[0].set_title('Real Image')
          #     axs[1].imshow(cv2.cvtColor(loaded_dictionary_images_sim[dataset_index][i], cv2.COLOR_BGR2RGB))
          #     axs[1].set_title('Sim Image')
          img=loaded_dictionary_images_real[dataset_index][i].copy()
          # index,label, x_min, y_min, x_max, y_max
    
          if task_type=="kitti":
              if i in boxes_real_dict[dataset_index]:
                  for label, x_min, y_min, x_max, y_max in boxes_real_dict[dataset_index][i]:
                      if label==3:
                          label_text="Car"
                      else:
                          label_text="Other"
                      if x_max > x_min and y_max > y_min:
                          cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                          cv2.putText(img, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          
              axs[0,2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
              axs[0,2].set_title('Real Image + bounding')
    
    
          
          # if task_type=="kitti":
          binary_mask_real = (loaded_semantic_id_real[dataset_index][i] == road).astype(np.uint8)
          binary_mask_fake = (loaded_semantic_id_sim[dataset_index][i] == road).astype(np.uint8)
    
    
          diff_mask_road=abs(binary_mask_fake-binary_mask_real)
    
          axs[1, 0].imshow(binary_mask_real, cmap='gray')
          axs[1, 0].set_title('Real Road Mask')
    
          axs[1, 1].imshow(binary_mask_fake, cmap='gray')
          axs[1, 1].set_title('Sim Road Mask')
    
          axs[1, 2].imshow(diff_mask_road, cmap='gray')
          axs[1, 2].set_title('Diff Road Mask')

          if task_type=="kitti":
            binary_mask_real = (loaded_semantic_id_real[dataset_index][i] == additional_id_init).astype(np.uint8)
            binary_mask_fake = (loaded_semantic_id_sim[dataset_index][i] == additional_id_init).astype(np.uint8)
        
            
            diff_mask_road=abs(binary_mask_fake-binary_mask_real)
        
            axs[2, 0].imshow(binary_mask_real, cmap='gray')
            axs[2, 0].set_title('Real Car Mask')
        
            axs[2, 1].imshow(binary_mask_fake, cmap='gray')
            axs[2, 1].set_title('Sim Car Mask')
        
            axs[2, 2].imshow(diff_mask_road, cmap='gray')
            axs[2, 2].set_title('Diff Car Mask')
    
          for ax in axs.flat:
              ax.set_xticks([])
              ax.set_yticks([])
    
          # Show the plot for the current iteration
          plt.show()


def save_sim_real_outputs(
    task_type,
    segmentation_model,
    real_path,
    sim_path,
    additional_id,
    height,
    width,
    dataset_index_list_test,
    test_indexes_gan,
    loaded_dictionary_images_real,
    loaded_dictionary_images_sim
):
    """
    The function `save_sim_real_outputs` saves simulated and real outputs as images and calculates the
    error between them.
    
    :param task_type: The type of task being performed, such as "kitti" or "donkey"
    :param segmentation_model: The `segmentation_model` parameter is the model used for image
    segmentation. It takes an input image and predicts the segmentation masks for different classes in
    the image
    :param real_path: The `real_path` parameter is a string that represents the path where the real
    images will be saved
    :param sim_path: The `sim_path` parameter is a string that represents the path where the simulated
    images will be saved
    :param additional_id: The `additional_id` parameter is an identifier for a specific class or
    category in the segmentation model. It is used to create binary masks for this specific class in the
    output images
    :param height: The height of the images in pixels
    :param width: The `width` parameter represents the width of the images in pixels
    :param dataset_index_list_test: A list of dataset indexes to iterate over for testing
    :param test_indexes_gan: A list of indexes representing the test samples for each dataset. Each
    element in the list corresponds to a dataset, and contains a list of indexes for the test samples in
    that dataset
    :param loaded_dictionary_images_real: A dictionary containing the loaded real images for each
    dataset index and test index
    :param loaded_dictionary_images_sim: A dictionary containing the loaded simulated images. The keys
    of the dictionary represent the dataset index, and the values are lists of images
    """
    for dataset_index in dataset_index_list_test:
          print(dataset_index)
          counter=0
          for i in test_indexes_gan[dataset_index]:
                if i%50==0:
                    print(i," out of ",len(test_indexes_gan[dataset_index]))
                counter+=1
                input_image_real = np.array(loaded_dictionary_images_real[dataset_index][i])
                input_image_real = tf.convert_to_tensor(input_image_real.astype(np.uint8), np.uint8)
                input_image_real = tf.reverse(input_image_real, axis=[-1])
                
                png_image = tf.image.encode_png(input_image_real)
                os.makedirs('./'+task_type+'/content/output_plots/real/'+real_path+'/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/real/'+real_path+'/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())
                input_image_real = tf.cast(input_image_real, np.float16)
                input_image_real = tf.transpose(input_image_real, (2, 0, 1))
        
                input_image_sim = np.array(loaded_dictionary_images_sim[dataset_index][i])
                input_image_sim_for_pix2pix=input_image_sim
                input_image_sim = tf.convert_to_tensor(input_image_sim.astype(np.uint8), np.uint8)
                input_image_sim = tf.reverse(input_image_sim, axis=[-1])
                png_image = tf.image.encode_png(tf.cast(input_image_sim,np.uint8))
                os.makedirs('./'+task_type+'/content/output_plots/sim/'+sim_path+'/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/sim/'+sim_path+'/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())
                input_image_sim = tf.cast(input_image_sim, np.float16)
                input_image_sim = tf.transpose(input_image_sim, (2, 0, 1))
                
                colored_mask = np.zeros_like(input_image_sim_for_pix2pix)
                input_image_real=tf.expand_dims(input_image_real, 0)
                
                pred_masks =segmentation_model.predict(input_image_real).logits
                created_mask=cast_to_int32(create_mask(pred_masks))
                created_mask_real = tf.image.resize(created_mask, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                created_mask_real = tf.cast(created_mask_real, np.uint8)

                input_image_sim=tf.expand_dims(input_image_sim, 0)
                pred_masks =segmentation_model.predict(input_image_sim).logits
                created_mask=cast_to_int32(create_mask(pred_masks))
                created_mask_sim = tf.image.resize(created_mask, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                created_mask_sim = tf.cast(created_mask_sim, np.uint8)

                if task_type=="kitti":
                    vectorized_map_additional=np.vectorize(map_values_car)
                    
                elif task_type=="donkey":
                    vectorized_map_additional=np.vectorize(map_values_donkey)

                real_mask=vectorized_map_additional(created_mask_real)
                sim_mask=vectorized_map_additional(created_mask_sim)
              
                colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(created_mask_real))
                colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(created_mask_real))
                colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(created_mask_real))
                png_image = tf.image.encode_png(colored_mask)
                os.makedirs('./'+task_type+'/content/output_plots/real/'+real_path+'_mask/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/real/'+real_path+'_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())

                if task_type=="donkey":
                    binary_mask_real1 = (real_mask == 1).astype(np.uint8)
                    binary_mask_real2 = (real_mask == 2).astype(np.uint8)
                    binary_mask_real = binary_mask_real1+binary_mask_real2
                else:
                    binary_mask_real = (real_mask == additional_id).astype(np.uint8)

                colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(binary_mask_real))
                colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(binary_mask_real))
                colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(binary_mask_real))
                png_image = tf.image.encode_png(colored_mask)
                os.makedirs('./'+task_type+'/content/output_plots/real/'+real_path+'_additional_mask/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/real/'+real_path+'_additional_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())

                colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(created_mask_sim))
                colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(created_mask_sim))
                colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(created_mask_sim))
                png_image = tf.image.encode_png(colored_mask)
                os.makedirs('./'+task_type+'/content/output_plots/sim/'+sim_path+'_mask/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/sim/'+sim_path+'_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())
              
                if task_type=="donkey":
                    binary_mask_sim1 = (sim_mask == 1).astype(np.uint8)
                    binary_mask_sim2 = (sim_mask == 2).astype(np.uint8)
                    binary_mask_sim = binary_mask_sim1+binary_mask_sim2
                else:
                    binary_mask_sim = (sim_mask == additional_id).astype(np.uint8)
                colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(binary_mask_sim))
                colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(binary_mask_sim))
                colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(binary_mask_sim))
                png_image = tf.image.encode_png(colored_mask)
                os.makedirs('./'+task_type+'/content/output_plots/sim/'+sim_path+'_additional_mask/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/sim/'+sim_path+'_additional_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())
                
                real_mask = np.array(real_mask)
                real_mask = tf.convert_to_tensor(real_mask.astype(np.float16), np.float16)
    
                sim_mask = np.array(sim_mask)
                sim_mask = tf.convert_to_tensor(sim_mask.astype(np.float16), np.float16)

                binary_mask_real = np.array(binary_mask_real)
                binary_mask_real = tf.convert_to_tensor(binary_mask_real.astype(np.float16), np.float16)

                binary_mask_sim = np.array(binary_mask_sim)
                binary_mask_sim = tf.convert_to_tensor(binary_mask_sim.astype(np.float16), np.float16)
    
                real_mask=cast_to_int32(real_mask)
                sim_mask=cast_to_int32(sim_mask)
                error_sim_real=abs(sim_mask-real_mask)
                error_sim_real_car=abs(binary_mask_sim-binary_mask_real)

                error_data_sim = {
                    "sim_real": np.sum(error_sim_real) / (height * width),
                    "additional": np.sum(error_sim_real_car) / (height * width)
                }

                output_folder = './'+task_type+'/content/output_plots/sim/'+sim_path+'_mask_error/'
                os.makedirs(output_folder, exist_ok=True)
                output_path = output_folder + str(dataset_index) + "_" + str(i) +".json"
                with open(output_path, "w") as json_file:
                    json.dump(error_data_sim, json_file, indent=4)


def save_cyclegan_outputs(
    task_type,
    segmentation_model,
    generator_cyclegan,
    cyclegan_name,
    additional_id,
    height,
    width,
    dataset_index_list_test,
    test_indexes_gan,
    loaded_dictionary_images_real,
    loaded_dictionary_images_sim
):
    """
    The function `save_cyclegan_outputs` saves the outputs of a CycleGAN model, including generated
    images and masks, to the specified directories.
    
    :param task_type: The type of task being performed, such as "kitti" or "donkey"
    :param segmentation_model: The `segmentation_model` parameter is the model used for image
    segmentation. It takes an input image and predicts the segmentation masks for different classes in
    the image
    :param generator_cyclegan: The generator model of the CycleGAN
    :param cyclegan_name: The name of the CycleGAN model being used for saving the outputs
    :param additional_id: The `additional_id` parameter is an integer value that represents the
    additional class label in the segmentation model. It is used to create binary masks for the
    additional class in the generated images
    :param height: The height of the images in pixels
    :param width: The `width` parameter represents the desired width of the output images
    :param dataset_index_list_test: A list of dataset indexes for which the outputs need to be saved.
    Each dataset index corresponds to a specific dataset in the loaded_dictionary_images_real and
    loaded_dictionary_images_sim
    :param test_indexes_gan: test_indexes_gan is a dictionary that contains the indexes of the test
    images for each dataset. The keys of the dictionary represent the dataset index, and the values are
    lists of indexes
    :param loaded_dictionary_images_real: A dictionary containing the real images for each dataset index
    and image index
    :param loaded_dictionary_images_sim: The `loaded_dictionary_images_sim` parameter is a dictionary
    that contains the loaded images for the simulation dataset. It is structured as follows:
    """
    for dataset_index in dataset_index_list_test:
          counter=0
          for i in test_indexes_gan[dataset_index]:
                if i%50==0:
                    print(i," out of ",len(test_indexes_gan[dataset_index]))
                counter+=1
                input_image_real = np.array(loaded_dictionary_images_real[dataset_index][i])
                input_image_real = tf.convert_to_tensor(input_image_real.astype(np.uint8), np.uint8)
                input_image_real = tf.reverse(input_image_real, axis=[-1])
                

                input_image_real = tf.cast(input_image_real, np.float16)
                input_image_real = tf.transpose(input_image_real, (2, 0, 1))
        
                input_image_sim = np.array(loaded_dictionary_images_sim[dataset_index][i])
                input_image_sim_for_pix2pix=input_image_sim
                input_image_sim = tf.convert_to_tensor(input_image_sim.astype(np.uint8), np.uint8)
                input_image_sim = tf.reverse(input_image_sim, axis=[-1])
                input_image_sim = tf.cast(input_image_sim, np.float16)
                input_image_sim = tf.transpose(input_image_sim, (2, 0, 1))
                
                vectorized_map_car=np.vectorize(map_values_car)
        
                input_image_sim_for_cyclegan=tf.expand_dims(input_image_sim, 0)
                input_image_sim_for_cyclegan={'pixel_values': input_image_sim_for_cyclegan }
              
                fake_real_cycle = generate_images_cyclegan(generator_cyclegan, input_image_sim_for_cyclegan,False,height,width)
                fake_real_cycle = tf.image.resize(fake_real_cycle, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                fake_real_cycle=fake_real_cycle * (255/2) + (255/2)
                fake_real_cycle = tf.cast(fake_real_cycle, np.uint8)
                png_image = tf.image.encode_png(fake_real_cycle)
                os.makedirs('./'+task_type+'/content/output_plots/cyclegan/'+cyclegan_name+'/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/cyclegan/'+cyclegan_name+'/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())
                fake_real_cycle = tf.cast(fake_real_cycle, np.float16)
                fake_real_cycle = tf.transpose(fake_real_cycle, (2, 0, 1))
              
                input_image_real=tf.expand_dims(input_image_real, 0)
                pred_masks =segmentation_model.predict(input_image_real).logits
                created_mask=cast_to_int32(create_mask(pred_masks))
                created_mask_real = tf.image.resize(created_mask, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                created_mask_real = tf.cast(created_mask_real, np.uint8)
                real_label_value =vectorized_map_car(created_mask_real)
                binary_mask_real = (real_label_value == additional_id).astype(np.uint8)
                if task_type == "donkey":
                    binary_mask_real = binary_mask_real + (real_label_value == 2).astype(np.uint8)
              
                fake_real_cycle=tf.expand_dims(fake_real_cycle, 0)
                pred_masks =segmentation_model.predict(fake_real_cycle).logits
                created_mask=cast_to_int32(create_mask(pred_masks))
                created_mask_cyclegan = tf.image.resize(created_mask, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                created_mask_cyclegan = tf.cast(created_mask_cyclegan, np.uint8)
                cyclegan_label_value =vectorized_map_car(created_mask_cyclegan)
                
                binary_mask_cyclegan = (cyclegan_label_value == additional_id).astype(np.uint8)
                if task_type == "donkey":
                    binary_mask_cyclegan = binary_mask_cyclegan + (cyclegan_label_value == 2).astype(np.uint8)
                colored_mask = np.zeros_like(input_image_sim_for_pix2pix)
                colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(binary_mask_cyclegan))
                colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(binary_mask_cyclegan))
                colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(binary_mask_cyclegan))
                png_image = tf.image.encode_png(colored_mask)
                os.makedirs('./'+task_type+'/content/output_plots/cyclegan/'+cyclegan_name+'_additional_mask/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/cyclegan/'+cyclegan_name+'_additional_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())

                colored_mask = np.zeros_like(input_image_sim_for_pix2pix)
                colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(created_mask_cyclegan))
                colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(created_mask_cyclegan))
                colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(created_mask_cyclegan))
                png_image = tf.image.encode_png(colored_mask)
                os.makedirs('./'+task_type+'/content/output_plots/cyclegan/'+cyclegan_name+'_mask/', exist_ok=True)
                with open('./'+task_type+'/content/output_plots/cyclegan/'+cyclegan_name+'_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                  f.write(png_image.numpy())
                
                input_image_sim=tf.expand_dims(input_image_sim, 0)
                pred_masks =segmentation_model.predict(input_image_sim).logits
                created_mask=cast_to_int32(create_mask(pred_masks))
                created_mask_sim = tf.image.resize(created_mask, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                created_mask_sim = tf.cast(created_mask_sim, np.uint8)

                if task_type=="kitti":
                    vectorized_map=np.vectorize(map_values_kitti)
                    
                elif task_type=="donkey":
                    vectorized_map=np.vectorize(map_values_donkey)
                
                created_mask_cyclegan=cast_to_int32(created_mask_cyclegan)
                created_mask_cyclegan=np.squeeze(created_mask_cyclegan)

                binary_mask_cyclegan=cast_to_int32(binary_mask_cyclegan)
                binary_mask_cyclegan=np.squeeze(binary_mask_cyclegan)

                binary_mask_real=cast_to_int32(binary_mask_real)
                binary_mask_real=np.squeeze(binary_mask_real)

                created_mask_real=cast_to_int32(created_mask_real)
                created_mask_real=np.squeeze(created_mask_real)
                created_mask_sim=cast_to_int32(created_mask_sim)
                created_mask_sim=np.squeeze(created_mask_sim)

                error_sim_real=abs(created_mask_sim-created_mask_real)
                error_cyclegan_sim=abs(created_mask_cyclegan-created_mask_sim)
                error_cyclegan_real=abs(created_mask_cyclegan-created_mask_real)
                error_cyclegan_car=abs(binary_mask_cyclegan-binary_mask_real)
    
                error_data_cyclegan = {
                    "sim_real": np.sum(error_sim_real) / (height * width),
                    "sim": np.sum(error_cyclegan_sim) / (height * width),
                    "real": np.sum(error_cyclegan_real) / (height * width),
                    "additional": np.sum(error_cyclegan_car) / (height * width)
                }

                output_folder = './'+task_type+'/content/output_plots/cyclegan/'+cyclegan_name+'_mask_error/'
                os.makedirs(output_folder, exist_ok=True)
                output_path = output_folder + str(dataset_index) + "_" + str(i) +".json"
                with open(output_path, "w") as json_file:
                    json.dump(error_data_cyclegan, json_file, indent=4)


def save_pix2pix_mask_outputs(task_type,limit,segmentation_model,two_classes,generator_pix2pix_mask,pix2pix_mask_name,additional_id,height,width,input_domain,pix2pix_mask_type,dataset_index_list_test,test_indexes_gan,loaded_dictionary_images_real,loaded_dictionary_images_sim,loaded_semantic_id_real,loaded_semantic_id_sim):
    """
    The function `save_pix2pix_mask_outputs` saves the outputs of a Pix2Pix model for mask generation,
    including the generated masks, error metrics, and visualization images.
    
    :param task_type: The type of task being performed (e.g., "kitti" or "donkey")
    :param limit: The `limit` parameter determines the maximum number of iterations or loops that the
    code will run. It is used to control the number of times the code will execute a certain block of
    code or perform a certain task
    :param segmentation_model: A segmentation model that is used to generate masks for input images
    :param two_classes: The `two_classes` parameter is a boolean value that determines whether the task
    involves two classes or not
    :param generator_pix2pix_mask: The generator_pix2pix_mask is a generator model used in the Pix2Pix
    network for generating mask outputs
    :param pix2pix_mask_name: The `pix2pix_mask_name` parameter is a string that represents the name of
    the pix2pix mask. It is used to create the output folder and file names for saving the generated
    masks and error data
    :param additional_id: The additional_id parameter is the value that represents the additional class
    in the segmentation mask. This is used to create a binary mask for the additional class in the
    pix2pix mask outputs
    :param height: The height parameter represents the desired height of the output images
    :param width: The width of the input images
    :param input_domain: The input_domain parameter specifies whether the input images are from the
    "real" domain or the "sim" domain
    :param pix2pix_mask_type: The `pix2pix_mask_type` parameter is a string that specifies the type of
    mask used in the Pix2Pix model. It can have two possible values: "manual" or "automatic"
    :param dataset_index_list_test: A list of dataset indexes to iterate over for testing
    :param test_indexes_gan: test_indexes_gan is a dictionary that contains the indexes of the test
    samples for each dataset. The keys of the dictionary are the dataset indexes, and the values are
    lists of indexes
    :param loaded_dictionary_images_real: A dictionary containing the loaded real images for each
    dataset index and image index
    :param loaded_dictionary_images_sim: loaded_dictionary_images_sim is a dictionary that contains the
    loaded images for the simulation domain. The keys of the dictionary represent the dataset index, and
    the values are lists of images for each dataset index
    :param loaded_semantic_id_real: loaded_semantic_id_real is a dictionary that contains the loaded
    semantic IDs for the real images in the dataset. The keys of the dictionary represent the dataset
    index, and the values are lists of semantic IDs corresponding to each image in the dataset
    :param loaded_semantic_id_sim: loaded_semantic_id_sim is a dictionary containing the semantic
    segmentation labels for the simulated images in the dataset. The keys of the dictionary represent
    the dataset index, and the values are lists of semantic segmentation labels for each image in the
    dataset
    """
    for dataset_index in dataset_index_list_test:
          print(dataset_index)
          counter=0
          for i in test_indexes_gan[dataset_index]:
                if i%50==0:
                    print(i," out of ",len(test_indexes_gan[dataset_index]))
                counter+=1
                input_image_real = np.array(loaded_dictionary_images_real[dataset_index][i])
                input_image_real = tf.convert_to_tensor(input_image_real.astype(np.uint8), np.uint8)
                input_image_real = tf.reverse(input_image_real, axis=[-1])
                
                
                input_image_real = tf.cast(input_image_real, np.float16)
                input_image_real = tf.transpose(input_image_real, (2, 0, 1))
        
                input_image_sim = np.array(loaded_dictionary_images_sim[dataset_index][i])
                input_image_sim_for_pix2pix=input_image_sim
                input_image_sim = tf.convert_to_tensor(input_image_sim.astype(np.uint8), np.uint8)
                input_image_sim = tf.reverse(input_image_sim, axis=[-1])
                
                input_image_sim = tf.cast(input_image_sim, np.float16)
              
                input_image_sim = tf.transpose(input_image_sim, (2, 0, 1))
                
                vectorized_map_car=np.vectorize(map_values_car)
        
                colored_mask = np.zeros_like(input_image_sim_for_pix2pix)
                colored_mask2 = np.zeros_like(input_image_sim_for_pix2pix)

                input_image_real=tf.expand_dims(input_image_real, 0)
                pred_masks =segmentation_model.predict(input_image_real).logits
                created_mask=cast_to_int32(create_mask(pred_masks))
                created_mask_real = tf.image.resize(created_mask, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                created_mask_real = tf.cast(created_mask_real, np.uint8)

                real_label_value =vectorized_map_car(created_mask_real)
                binary_mask_real = (real_label_value == additional_id).astype(np.uint8)
                if task_type == "donkey":
                    binary_mask_real = binary_mask_real + (real_label_value == 2).astype(np.uint8)

                input_image_sim=tf.expand_dims(input_image_sim, 0)
                pred_masks =segmentation_model.predict(input_image_sim).logits
                created_mask=cast_to_int32(create_mask(pred_masks))
                created_mask_sim = tf.image.resize(created_mask, [height,width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                created_mask_sim = tf.cast(created_mask_sim, np.uint8)

                if pix2pix_mask_type=="manual": 
                    if (input_domain=="real"):
                        mask_in=loaded_semantic_id_real[dataset_index][i]
                    else:
                        mask_in=loaded_semantic_id_sim[dataset_index][i]
                    
                else:
                    if (input_domain=="real"):
                        mask_in=created_mask_real[:,:,0]
                    else:
                        mask_in=created_mask_sim[:,:,0]
              
              
                if (input_domain=="real"):
                    colored_mask[:, :, 0] = np.vectorize(map_to_b)(mask_in)
                    colored_mask[:, :, 1] = np.vectorize(map_to_g)(mask_in)
                    colored_mask[:, :, 2] = np.vectorize(map_to_r)(mask_in)
                    input_image_real_for_pix2pix_mask=colored_mask
                    input_image_real_for_pix2pix_mask = tf.convert_to_tensor(input_image_real_for_pix2pix_mask.astype(np.float16), np.float16)
                    input_image_real_for_pix2pix_mask = tf.reverse(input_image_real_for_pix2pix_mask, axis=[-1])
                    input_image_real_for_pix2pix_mask = tf.image.resize(input_image_real_for_pix2pix_mask, [512,512],
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    input_image_real_for_pix2pix_mask=tf.expand_dims(input_image_real_for_pix2pix_mask, 0)
                    input_image_real_for_pix2pix_mask = (input_image_real_for_pix2pix_mask / 127.5) - 1
                    fake_real_pix2pix_mask_real = generate_images_pix2pix(generator_pix2pix_mask, input_image_real_for_pix2pix_mask,False)
                    fake_real_pix2pix_mask_real = tf.image.resize(fake_real_pix2pix_mask_real, [height,width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    fake_real_pix2pix_mask_real=fake_real_pix2pix_mask_real * (255/2) + (255/2)
                    fake_real_pix2pix_mask_real = tf.cast(fake_real_pix2pix_mask_real, np.uint8)
                    png_image = tf.image.encode_png(fake_real_pix2pix_mask_real)
                    os.makedirs('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_real/', exist_ok=True)
                    with open('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_real/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                        f.write(png_image.numpy())
                    fake_real_pix2pix_mask_real = tf.cast(fake_real_pix2pix_mask_real, np.float16)
                    fake_real_pix2pix_mask_real = tf.transpose(fake_real_pix2pix_mask_real, (2, 0, 1))


                    fake_real_pix2pix_mask_real=tf.expand_dims(fake_real_pix2pix_mask_real, 0)
                    pred_masks = segmentation_model.predict(fake_real_pix2pix_mask_real).logits
                    created_mask=cast_to_int32(create_mask(pred_masks))
                    created_mask_pix2pix_mask_real = tf.image.resize(created_mask, [height,width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


                    pix2pix_label_value=created_mask_pix2pix_mask_real
                    pix2pix_label_value =vectorized_map_car(pix2pix_label_value)

                    
                    
                    binary_mask_pix2pix = (pix2pix_label_value == additional_id).astype(np.uint8)
                    if task_type == "donkey":
                        binary_mask_pix2pix = binary_mask_pix2pix + (pix2pix_label_value == 2).astype(np.uint8)
                    colored_mask = np.zeros_like(input_image_sim_for_pix2pix)
                    colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(binary_mask_pix2pix))
                    colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(binary_mask_pix2pix))
                    colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(binary_mask_pix2pix))
                    png_image = tf.image.encode_png(colored_mask)
                    os.makedirs('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_real_additional_mask/', exist_ok=True)
                    with open('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_real_additional_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                      f.write(png_image.numpy())
                    
                    created_mask_pix2pix_mask_real = tf.cast(created_mask_pix2pix_mask_real, np.uint8)
                    colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(created_mask_pix2pix_mask_real))
                    colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(created_mask_pix2pix_mask_real))
                    colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(created_mask_pix2pix_mask_real))
                    png_image = tf.image.encode_png(colored_mask)
                    os.makedirs('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_real_mask/', exist_ok=True)
                    with open('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_real_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                        f.write(png_image.numpy())
                    created_mask_pix2pix_mask_real=cast_to_int32(created_mask_pix2pix_mask_real)
                    created_mask_pix2pix_mask_real=np.squeeze(created_mask_pix2pix_mask_real)
                    

              
                else:
                    colored_mask2[:, :, 0] = np.vectorize(map_to_b)(mask_in)
                    colored_mask2[:, :, 1] = np.vectorize(map_to_g)(mask_in)
                    colored_mask2[:, :, 2] = np.vectorize(map_to_r)(mask_in)
                    input_image_sim_for_pix2pix_mask=colored_mask2
                    input_image_sim_for_pix2pix_mask = tf.convert_to_tensor(input_image_sim_for_pix2pix_mask.astype(np.float16), np.float16)
                    input_image_sim_for_pix2pix_mask = tf.reverse(input_image_sim_for_pix2pix_mask, axis=[-1])
                    input_image_sim_for_pix2pix_mask = tf.image.resize(input_image_sim_for_pix2pix_mask, [512,512],
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    input_image_sim_for_pix2pix_mask=tf.expand_dims(input_image_sim_for_pix2pix_mask, 0)
                    input_image_sim_for_pix2pix_mask = (input_image_sim_for_pix2pix_mask / 127.5) - 1  
                    fake_real_pix2pix_mask_sim = generate_images_pix2pix(generator_pix2pix_mask, input_image_sim_for_pix2pix_mask,False)
                    fake_real_pix2pix_mask_sim = tf.image.resize(fake_real_pix2pix_mask_sim, [height,width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    fake_real_pix2pix_mask_sim=fake_real_pix2pix_mask_sim * (255/2) + (255/2)
                    fake_real_pix2pix_mask_sim = tf.cast(fake_real_pix2pix_mask_sim, np.uint8)
                    png_image = tf.image.encode_png(fake_real_pix2pix_mask_sim)
                    os.makedirs('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_sim/', exist_ok=True)
                    with open('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_sim/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                      f.write(png_image.numpy())
                    fake_real_pix2pix_mask_sim = tf.cast(fake_real_pix2pix_mask_sim, np.float16)
                    fake_real_pix2pix_mask_sim = tf.transpose(fake_real_pix2pix_mask_sim, (2, 0, 1))

                    fake_real_pix2pix_mask_sim=tf.expand_dims(fake_real_pix2pix_mask_sim, 0)
                    pred_masks = segmentation_model.predict(fake_real_pix2pix_mask_sim).logits
                    created_mask=cast_to_int32(create_mask(pred_masks))
                    created_mask_pix2pix_mask_sim = tf.image.resize(created_mask, [height,width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    
                    
                    pix2pix_label_value=created_mask_pix2pix_mask_sim
                    pix2pix_label_value =vectorized_map_car(pix2pix_label_value)
                    binary_mask_pix2pix = (pix2pix_label_value == additional_id).astype(np.uint8)
                    if task_type == "donkey":
                        binary_mask_pix2pix = binary_mask_pix2pix + (pix2pix_label_value == 2).astype(np.uint8)
                    colored_mask = np.zeros_like(input_image_sim_for_pix2pix)
                    colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(binary_mask_pix2pix))
                    colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(binary_mask_pix2pix))
                    colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(binary_mask_pix2pix))
                    png_image = tf.image.encode_png(colored_mask)
                    os.makedirs('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_sim_additional_mask/', exist_ok=True)
                    with open('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_sim_additional_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                      f.write(png_image.numpy())
                    
                    created_mask_pix2pix_mask_sim = tf.cast(created_mask_pix2pix_mask_sim, np.uint8)
                    colored_mask[:, :, 0] = np.vectorize(map_to_b)(tf.squeeze(created_mask_pix2pix_mask_sim))
                    colored_mask[:, :, 1] = np.vectorize(map_to_g)(tf.squeeze(created_mask_pix2pix_mask_sim))
                    colored_mask[:, :, 2] = np.vectorize(map_to_r)(tf.squeeze(created_mask_pix2pix_mask_sim))
                    png_image = tf.image.encode_png(colored_mask)
                    os.makedirs('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_sim_mask/', exist_ok=True)
                    with open('./'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_sim_mask/'+str(dataset_index)+'_'+str(i)+'.png', 'wb') as f:
                      f.write(png_image.numpy())
                    created_mask_pix2pix_mask_sim=cast_to_int32(created_mask_pix2pix_mask_sim)
                    created_mask_pix2pix_mask_sim=np.squeeze(created_mask_pix2pix_mask_sim)
                    
                
                if task_type=="kitti":
                    vectorized_map=np.vectorize(map_values_car)
                elif task_type=="donkey":
                    vectorized_map=np.vectorize(map_values_donkey)
                real_label_value=created_mask_real

                
                binary_mask_pix2pix=cast_to_int32(binary_mask_pix2pix)
                binary_mask_pix2pix=np.squeeze(binary_mask_pix2pix)
                binary_mask_real=cast_to_int32(binary_mask_real)
                binary_mask_real=np.squeeze(binary_mask_real)

                created_mask_real=cast_to_int32(created_mask_real)
                created_mask_real=np.squeeze(created_mask_real)
                created_mask_sim=cast_to_int32(created_mask_sim)
                created_mask_sim=np.squeeze(created_mask_sim)

                error_pix2pix_car=abs(binary_mask_pix2pix-binary_mask_real)
                error_sim_real=abs(created_mask_sim-created_mask_real)

                if (input_domain=="real"):

                    error_pix2pix_sim_mask_real=abs(created_mask_pix2pix_mask_real-created_mask_sim)
                    error_pix2pix_real_mask_real=abs(created_mask_pix2pix_mask_real-created_mask_real)
                    
                    error_data_pix2pix_mask_real = {
                        "sim_real": np.sum(error_sim_real) / (width * height),
                        "sim": np.sum(error_pix2pix_sim_mask_real) / (width * height),
                        "real": np.sum(error_pix2pix_real_mask_real) / (width * height),
                        "additional": np.sum(error_pix2pix_car) / (width * height)
                    }
    
                    output_folder = './'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_real_mask_error/'
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = output_folder + str(dataset_index) + "_" + str(i) +".json"
                    with open(output_path, "w") as json_file:
                        json.dump(error_data_pix2pix_mask_real, json_file, indent=4)
                    
                else:
                    error_pix2pix_sim_mask_sim=abs(created_mask_pix2pix_mask_sim-created_mask_sim)
                    error_pix2pix_real_mask_sim=abs(created_mask_pix2pix_mask_sim-created_mask_real)
                    error_data_pix2pix_mask_sim = {
                        "sim_real": np.sum(error_sim_real) / (width * height),
                        "sim": np.sum(error_pix2pix_sim_mask_sim) / (width * height),
                        "real": np.sum(error_pix2pix_real_mask_sim) / (width * height),
                        "additional": np.sum(error_pix2pix_car) / (width * height)
                    }
                    output_folder = './'+task_type+'/content/output_plots/pix2pix_mask_'+pix2pix_mask_type+'/'+pix2pix_mask_name+'_sim_mask_error/'
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = output_folder + str(dataset_index) + "_" + str(i) +".json"
                    with open(output_path, "w") as json_file:
                        json.dump(error_data_pix2pix_mask_sim, json_file, indent=4)