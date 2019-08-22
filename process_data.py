import os
import json
import collections
import string
import random

import progressbar
from PIL import Image
import numpy as np

import tensorflow as tf
from keras import models
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image


def clean_captions(all_captions):
    all_captions_cleaned = {}
    punt = str.maketrans('', '', string.punctuation)

    for key, capt_list in all_captions.items():
        capt_list_cleaned = []
        for i in range(len(capt_list)):
            cap = capt_list[i]

            cap = cap.split()

            # convert to lower case
            cap = [word.lower() for word in cap]
            # remove punctuation from each token

            cap = [w.translate(punt) for w in cap]

            cap = [word for word in cap if len(word) > 1]

            # desc = [word for word in desc if word.isalpha()]

            capt_list_cleaned.append(' '.join(cap))

        all_captions_cleaned[key] = capt_list_cleaned

    return all_captions_cleaned


def to_captions_list(all_captions):
    all_train_captions = []
    for key, val in all_captions.items():
        for c in val:
            all_train_captions.append(c)
    return all_train_captions


def add_start_end_token(all_captions_list):
    all_captions_list_token = []
    for c in all_captions_list:
        new_c = '<start> ' + c + ' <end>'
        all_captions_list_token.append(new_c)

    return all_captions_list_token


def generate_vocabulary(all_captions_list):
    vocab = []
    for c in all_captions_list:
        vocab = vocab + c.split()

    return vocab


def load_all_images_name(dataset_dir_path):
    return os.listdir(dataset_dir_path)




def preprocess_images(dataset_dir_path, train_images_name, output_file_name):
    """
    if (os.path.isfile(output_file_name)):

        with open(output_file_name) as f:
            images_as_vector = json.load(f)

    else:
    """
    os.chdir(dataset_dir_path)

    images_as_vector = collections.defaultdict()

    modelvgg = VGG16(include_top=True, weights=None)
    #       modelvgg.load_weights("../input/vgg16-weights-image-captioning/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    modelvgg.summary()

    with progressbar.ProgressBar(max_value=len(train_images_name)) as bar:
        for i, image_path in enumerate(train_images_name):
            img = image.load_img(image_path, target_size=(224, 224,3))

            im = image.img_to_array(img)


            im = preprocess_input(im)
            im_pred = modelvgg.predict(im.reshape((1,) + im.shape[:3]))



            images_as_vector[image_path] = im_pred.flatten()
            bar.update(i)

    return images_as_vector

"""
    with open(output_file_name, 'w') as outfile:
        outfile.write(json.dumps(images_as_vector)
"""
