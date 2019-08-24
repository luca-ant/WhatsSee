import collections
import json
import os
import random

import tensorflow as tf

from keras import Input, Model
from keras.layers import Dropout, Dense, LSTM, Embedding, add


def create_NN(vocab_size, max_cap_len):
    input_image = Input(shape=(4096,))
    fe1 = Dropout(0.5)(input_image)
    fe2 = Dense(256, activation='relu')(fe1)

    input_text = Input(shape=(max_cap_len,))
    se1 = Embedding(vocab_size, 64, mask_zero=True)(input_text)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[input_image, input_text], outputs=outputs)

    return model


class Dataset:

    @staticmethod
    def download_dataset(dataset, data_path):
        return dataset.download_dataset(data_path)

    @staticmethod
    def load_captions(dataset):
        return dataset.load_captions()

    @staticmethod
    def load_train_captions(dataset):
        return dataset.load_train_captions()

    @staticmethod
    def load_images_name(dataset, images_id_list):
        return dataset.load_images_name(images_id_list)

    @staticmethod
    def get_image_name(dataset, image_id):
        return dataset.get_image_name(image_id)

    @staticmethod
    def load_all_images_name(dataset):
        return dataset.load_all_images_name()


class FlickrDataset():
    caption_file_path = ""
    dataset_dir_path = ""

    def __init__(self):
        super().__init__()

    def download_dataset(self, data_path):
        os.chdir(data_path)

        name_of_zip = 'Flickr8k_text.zip'

        if not os.path.exists(os.path.abspath('.') + '/Flickr8k_text/'):

            caption_zip = tf.keras.utils.get_file(name_of_zip,
                                                  cache_subdir=os.path.abspath('.'),
                                                  origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                                                  extract=True)
            caption_file_path = os.path.dirname(caption_zip) + '/Flickr8k_text/Flickr8k.token.txt'


        else:
            caption_file_path = os.path.abspath('.') + '/Flickr8k_text/Flickr8k.token.txt'
            print("Captions already exists")

        name_of_zip = 'Flickr8k_Dataset.zip'

        if not os.path.exists(os.path.abspath('.') + '/Flickr8k_Dataset/Flicker8k_Dataset/'):

            image_zip = tf.keras.utils.get_file(name_of_zip,
                                                cache_subdir=os.path.abspath("."),
                                                origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                                                extract=True)
            dataset_dir_path = os.path.dirname(image_zip) + '/Flickr8k_Dataset/Flicker8k_Dataset/'


        else:
            dataset_dir_path = os.path.abspath('.') + '/Flickr8k_Dataset/Flicker8k_Dataset/'
            print("Images dataset already exists")

        os.chdir("..")

        self.caption_file_path = os.path.abspath(caption_file_path)
        self.dataset_dir_path = os.path.abspath(dataset_dir_path)
        return os.path.abspath(caption_file_path), os.path.abspath(dataset_dir_path)

    def load_captions(self):
        all_captions = collections.defaultdict(list)

        with open(self.caption_file_path, 'r') as f:

            for line in f:
                tokens = line.split()
                if len(line) < 2:
                    continue
                # take the first token as the image id, the rest as the description
                image_id, image_cap = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                image_cap = ' '.join(image_cap)

                all_captions[image_id].append(image_cap)

        return all_captions

    def load_train_captions(self):
        train_captions = collections.defaultdict(list)
        image_names=[]
        with open(os.path.dirname(self.caption_file_path) + "/Flickr_8k.trainImages.txt", 'r') as f:
            for line in f:
                image_names.append(line.strip())

        with open(self.caption_file_path, 'r') as f:

            for line in f:
                tokens = line.split()
                # take the first token as the image id, the rest as the description
                image_id, image_cap = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                image_cap = ' '.join(image_cap)
                if(image_id+".jpg" in image_names):
                    train_captions[image_id].append(image_cap)

        from process_data import clean_captions
        train_captions = clean_captions(train_captions)

        #num_training_examples = 10 # DEBUG
        #train_captions = dict(list(train_captions.items())[:num_training_examples])

        return train_captions

    def load_images_name(self, images_id_list):
        images_name_list = []
        for id in images_id_list:
            image_name = id + '.jpg'
            if os.path.isfile(self.dataset_dir_path + "/" + image_name):
                images_name_list.append(image_name)

        return images_name_list

    def get_image_name(self, image_id):
        image_name = image_id + ".jpg"
        return image_name

    def load_all_images_name(self):
        return os.listdir(self.dataset_dir_path)


class COCODataset():
    caption_file_path = ""
    dataset_dir_path = ""

    def __init__(self):
        super().__init__()

    def download_dataset(self, data_path):

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
            print("Captions already exists")

        name_of_zip = 'train2014.zip'

        if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):

            image_zip = tf.keras.utils.get_file(name_of_zip,
                                                cache_subdir=os.path.abspath("."),
                                                origin='http://images.cocodataset.org/zips/train2014.zip',
                                                extract=True)
            dataset_dir_path = os.path.dirname(image_zip) + '/train2014/'


        else:
            dataset_dir_path = os.path.abspath('.') + '/train2014/'
            print("Images dataset already exists")

        os.chdir("..")

        self.caption_file_path = os.path.abspath(caption_file_path)
        self.dataset_dir_path = os.path.abspath(dataset_dir_path)
        return (os.path.abspath(caption_file_path), os.path.abspath(dataset_dir_path))

    def load_captions(self):
        all_captions = collections.defaultdict(list)
        with open(self.caption_file_path, 'r') as f:
            captions = json.load(f)

        for c in captions['annotations']:
            caption_string = c['caption']
            image_id = c['image_id']
            all_captions[image_id].append(caption_string)

        return all_captions

    def load_train_captions(self):

        all_captions = self.load_captions()  # dict image_id - caption

        # Shuffle captions
        l = list(all_captions.items())
        random.shuffle(l)
        all_captions = dict(l)

        num_training_examples = 8000  # DEBUG

        num_training_examples = 10  # DEBUG

        train_captions = dict(list(all_captions.items())[:num_training_examples])

        from process_data import clean_captions
        train_captions = clean_captions(train_captions)

        return train_captions

    def load_images_name(self, images_id_list):
        images_name_list = []
        for id in images_id_list:
            image_name = 'COCO_train2014_' + '%012d.jpg' % (id)
            if os.path.isfile(self.dataset_dir_path + "/" + image_name):
                images_name_list.append(image_name)

        return images_name_list

    def get_image_name(self, image_id):
        image_name = "COCO_train2014_" + "%012d.jpg" % (image_id)
        return image_name

    def load_all_images_name(self):
        return os.listdir(self.dataset_dir_path)
