import os
import json
import collections
import string
import random
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.applications.inception_v3 import preprocess_input

data_path = "./data/"

if not os.path.isdir(data_path):
    os.makedirs(data_path)


def download_dataset():
    os.chdir(data_path)

    name_of_zip = 'captions.zip'

    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):

        caption_zip = tf.keras.utils.get_file(name_of_zip,
                                              cache_subdir=os.path.abspath('.'),
                                              origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                              extract=True)
        caption_file_path = os.path.dirname(caption_zip) + '/annotations/captions_train2014.json'


    else:
        caption_file_path = os.path.abspath('.') + '/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'

    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                            cache_subdir=os.path.abspath("."),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        dataset_dir_path = os.path.dirname(image_zip) + '/train2014/'
    else:
        dataset_dir_path = os.path.abspath('.') + '/train2014/'

    os.chdir("..")

    return (os.path.abspath(caption_file_path), os.path.abspath(dataset_dir_path))


def load_captions(caption_file_path):
    all_captions = collections.defaultdict()
    with open(caption_file_path, 'r') as f:
        captions = json.load(f)

    for c in captions['annotations']:
        caption_string = '<start> ' + c['caption'] + ' <end>'
        image_id = c['image_id']

        all_captions[image_id].append(caption_string)

    return all_captions


def clean_captions(all_captions):
    punt = str.maketrans('', '', string.punctuation)

    for key, desc_list in all_captions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()

            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token

            desc = [w.translate(punt) for w in desc]

            desc = [word for word in desc if len(word) > 1]

            # desc = [word for word in desc if word.isalpha()]

            desc_list[i] = ' '.join(desc)


def load_images_name(dataset_dir_path):
    return os.listdir(dataset_dir_path)


def preprocess_images(dataset_dir_path, train_images_name, output_file_name):
    if (os.path.isfile(output_file_name)):

        with open(output_file_name) as f:
            images_as_vector = json.load(f)

    else:
        os.chdir(dataset_dir_path)
        # Get the InceptionV3 model trained on imagenet data
        model = InceptionV3(weights='imagenet')
        # Remove the last layer (output softmax layer) from the inception v3
        model_new = keras.Model(model.input, model.layers[-2].output)

        images_as_vector = collections.defaultdict()

        for image_path in train_images_name:
            # Convert all the images to size 299x299 as expected by the
            # inception v3 model
            img = Image.load_img(image_path, target_size=(299, 299))
            # Convert PIL image to numpy array of 3-dimensions
            i = Image.img_to_array(img)
            # Add one more dimension
            i = np.expand_dims(i, axis=0)
            # preprocess images using preprocess_input() from inception module
            i = preprocess_input(i)
            # reshape from (1, 2048) to (2048, )
            i = np.reshape(i, i.shape[1])
            images_as_vector[image_path] = i

        with open(output_file_name, 'w') as outfile:
            outfile.write(json.dumps(images_as_vector))

    return images_as_vector
