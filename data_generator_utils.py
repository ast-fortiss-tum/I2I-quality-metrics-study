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



OUTPUT_CHANNELS = 3
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_h5_to_dictionary(file_path):
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
    lowest_height = min(images_list1[0].shape[0], images_list2[0].shape[0])
    lowest_width = min(images_list1[0].shape[1], images_list2[0].shape[1])
    return lowest_height, lowest_width

def crop_images_to_lowest_dimensions(images_list, lowest_height, lowest_width):
    cropped_images = [image[:lowest_height, :lowest_width,:] for image in images_list]
    return cropped_images

def crop_1d_to_lowest_dimensions(images_list, lowest_height, lowest_width):
    cropped_images = [image[:lowest_height, :lowest_width] for image in images_list]
    return cropped_images
def display(display_list):
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
    return {'pixel_values': input_image, 'labels': output_image}

def map_values_kitti(value):
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
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]
    
def cast_to_int32(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.cast(image * 255, tf.int32)
    return image

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image
def random_jitter(image):

  image = tf.image.resize(image, [520, 520],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = random_crop(image)

  image = tf.image.random_flip_left_right(image)

  return image
def preprocess_image_train(input):
  image=input["pixel_values"]
  image = random_jitter(tf.transpose(image, (1, 2, 0)))
  image = normalize(image)
  return image
def preprocess_image_test(input):
  image=input["pixel_values"]
  image = random_jitter(tf.transpose(image, (1, 2, 0)))
  image = normalize(image)
  return image
def generate_images_cyclegan(model, test_input,plot,height,width):
  
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

  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
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
  prediction = model(test_input, training=True)
  

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']
  if plot:
      plt.figure(figsize=(15, 15))
      for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(tf.cast(display_list[i]* 0.5 + 0.5, tf.float32))
        plt.axis('off')
      plt.show()
  return prediction[0]


def map_values_car(value):
    return int(value)

def auto_canny(image, sigma=0.66):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged

counter=0


def map_to_r(number):
    colorr = [255, 128, 0, 0, 128, 0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 255, 128, 255, 0, 128]
    if number<20:
      return int(colorr[number])
    else:
      return 0

def map_to_g(number):

    colorg = [0, 128, 255, 128, 0, 0, 128, 0, 0, 128, 0, 128, 0, 128, 255, 128, 0, 128, 255, 128]

    if number<20:
      return int(colorg[number])
    else:
      return 0

def map_to_b(number):

    colorb = [0, 0, 0, 128, 128, 255, 255, 128, 0, 128, 128, 255, 128, 0, 0, 0, 128, 255, 255, 128]

    if number<20:
      return int(colorb[number])
    else:
      return 255

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_gan_indexes(dataset_index_list_test,loaded_dictionary_images_real,loaded_dictionary_images_sim,pattern):
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
    
          axs[0,0].imshow(cv2.cvtColor(loaded_dictionary_images_real[dataset_index][i], cv2.COLOR_BGR2RGB))
          axs[0,0].set_title('Real Image')
          axs[0,1].imshow(cv2.cvtColor(loaded_dictionary_images_sim[dataset_index][i], cv2.COLOR_BGR2RGB))
          axs[0,1].set_title('Sim Image')
          img=loaded_dictionary_images_real[dataset_index][i].copy()
    
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
          plt.show()