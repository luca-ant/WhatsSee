import collections
import json
import os
import random
import wget
import zipfile

from git import Repo, RemoteProgress
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


class Progress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if op_code == 32:
            print('Downloaded %d of %d, %s' % (cur_count, max_count, message))


class Dataset:

    @staticmethod
    def download_dataset(dataset):
        return dataset.download_dataset()

    @staticmethod
    def load_captions(dataset):
        return dataset.load_captions()

    @staticmethod
    def load_train_captions(dataset, num_training_examples):
        return dataset.load_train_captions(num_training_examples)

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

    def __init__(self, data_path):
        self.subdir = data_path + "flickr_dataset/"
        self.caption_dir_path = self.subdir + "captions/"
        self.images_dir_path = self.subdir + "images/"
        self.captions_file_path = self.caption_dir_path + 'Flickr8k.token.txt'

    def download_dataset(self):

        if not os.path.exists(self.caption_dir_path) or not os.path.exists(self.images_dir_path):
            print("DOWNLOADING FLICKR DATASET")
            os.removedirs(self.subdir)
            os.system("git clone --progress -v https://github.com/luca-ant/WhatsSee_dataset.git " + self.subdir)
            # Repo.clone_from("https://github.com/luca-ant/WhatsSee_dataset.git", self.data_path, progress=Progress())

        else:
            print("Captions already exists")
            print("Images dataset already exists")

        captions_file_path = self.caption_dir_path + 'Flickr8k.token.txt'
        images_dir_path = self.images_dir_path + ''

        return captions_file_path, images_dir_path

    def load_captions(self):
        all_captions = collections.defaultdict(list)

        with open(self.captions_file_path, 'r') as f:

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

    def load_train_captions(self, num_training_examples):
        train_captions = collections.defaultdict(list)
        image_names = []
        with open(os.path.dirname(self.captions_file_path) + "/Flickr_8k.trainImages.txt", 'r') as f:
            for line in f:
                image_names.append(line.strip())

        with open(self.captions_file_path, 'r') as f:

            for line in f:
                tokens = line.split()
                # take the first token as the image id, the rest as the description
                image_id, image_cap = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                image_cap = ' '.join(image_cap)
                if (image_id + ".jpg" in image_names):
                    train_captions[image_id].append(image_cap)

        if num_training_examples != 0:
            # Shuffle captions
            l = list(train_captions.items())
            random.shuffle(l)
            train_captions = dict(l)

            train_captions = dict(list(train_captions.items())[:num_training_examples])

        from process_data import clean_captions
        train_captions = clean_captions(train_captions)

        return train_captions

    def load_images_name(self, images_id_list):
        images_name_list = []
        for id in images_id_list:
            image_name = id + '.jpg'
            if os.path.isfile(self.images_dir_path + "/" + image_name):
                images_name_list.append(image_name)

        return images_name_list

    def get_image_name(self, image_id):
        image_name = image_id + ".jpg"
        return image_name

    def load_all_images_name(self):
        return os.listdir(self.images_dir_path)


class COCODataset():

    def __init__(self, data_path):
        self.subdir = data_path + "coco_dataset/"
        self.caption_dir_path = self.subdir + "captions/"
        self.images_dir_path = self.subdir + "images/"
        self.captions_file_path = self.caption_dir_path + 'annotations/captions_train2014.json'

    def download_dataset(self):

        name_of_zip = 'captions.zip'
        if not os.path.exists(self.caption_dir_path):
            os.makedirs(self.caption_dir_path, exist_ok=True)

            url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
            print("DOWNLOADING CAPTIONS FROM COCO DATASET")
            captions_zip = wget.download(url, self.subdir + name_of_zip)

            print("\nEXTRACTING ZIP CAPIOTN FILE INTO " + self.caption_dir_path)

            with zipfile.ZipFile(captions_zip, 'r') as zip:
                zip.extractall(self.caption_dir_path)

            os.remove(captions_zip)
            captions_file_path = self.caption_dir_path + 'annotations/captions_train2014.json'

        else:
            captions_file_path = self.caption_dir_path + 'annotations/captions_train2014.json'
            print("Captions already exists")

        name_of_zip = 'train2014.zip'

        if not os.path.exists(self.images_dir_path):

            os.makedirs(self.images_dir_path, exist_ok=True)

            url = 'http://images.cocodataset.org/zips/train2014.zip'
            print("DOWNLOADING IMAGES FROM COCO DATASET")
            images_zip = wget.download(url, self.subdir + name_of_zip)

            print("\nEXTRACTING ZIP IMAGES FILE INTO " + self.images_dir_path)

            with zipfile.ZipFile(images_zip, 'r') as zip:
                zip.extractall(self.images_dir_path)

            os.remove(images_zip)

            images_dir_path = self.images_dir_path + 'train2014/'



        else:
            images_dir_path = self.images_dir_path + 'train2014/'
            print("Images dataset already exists")

        return captions_file_path, images_dir_path

    def load_captions(self):
        all_captions = collections.defaultdict(list)
        with open(self.captions_file_path, 'r') as f:
            captions = json.load(f)

        for c in captions['annotations']:
            caption_string = c['caption']
            image_id = c['image_id']
            all_captions[image_id].append(caption_string)

        return all_captions

    def load_train_captions(self, num_training_examples):

        all_captions = self.load_captions()  # dict image_id - caption

        if num_training_examples == 0:
            train_captions = all_captions

        else:
            # Shuffle captions
            l = list(all_captions.items())
            random.shuffle(l)
            all_captions = dict(l)

            train_captions = dict(list(all_captions.items())[:num_training_examples])

        from process_data import clean_captions
        train_captions = clean_captions(train_captions)

        return train_captions

    def load_images_name(self, images_id_list):
        images_name_list = []
        for id in images_id_list:
            image_name = 'COCO_train2014_' + '%012d.jpg' % (id)
            if os.path.isfile(self.images_dir_path + "/" + image_name):
                images_name_list.append(image_name)

        return images_name_list

    def get_image_name(self, image_id):
        image_name = "COCO_train2014_" + "%012d.jpg" % (image_id)
        return image_name

    def load_all_images_name(self):
        return os.listdir(self.images_dir_path)
