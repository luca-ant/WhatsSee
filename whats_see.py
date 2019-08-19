import os
import json
import random

import tensorflow as tf
from tensorflow import keras

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


caption_file_path, dataset_dir_path = download_dataset()

print(caption_file_path, dataset_dir_path)

with open(caption_file_path, 'r') as f:
    captions = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_images_name = []

for c in captions['annotations']:
    caption_string = '<start> ' + c['caption'] + ' <end>'
    image_id = c['image_id']
    full_coco_image_path = dataset_dir_path + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_images_name.append(full_coco_image_path)
    all_captions.append(caption_string)

# Shuffle captions and image_names
train_captions, img_name_vector = random.shuffle(all_captions,
                                                 all_images_name,
                                                 random_state=1)

# Select the first 30000 captions from the shuffled set
num_training_examples = 30000
train_captions = train_captions[:num_training_examples]
train_images_name = img_name_vector[:num_training_examples]
