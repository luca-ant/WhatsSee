import collections
import json
import os
import random
import shutil

import wget
import zipfile
from git import Repo, RemoteProgress


class Progress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if op_code == 32:
            print('Downloaded %d of %d, %s' % (cur_count, max_count, message))


class Dataset:

    @staticmethod
    def download_dataset(dataset):
        return dataset.download_dataset()

    @staticmethod
    def get_image_name(dataset, image_id):
        return dataset.get_image_name(image_id)

    @staticmethod
    def create_dataset(dataset_name, data_dir):
        if dataset_name == "coco":
            dataset = COCODataset(data_dir)
        elif dataset_name == "flickr":
            dataset = FlickrDataset(data_dir)
        return dataset


class FlickrDataset():

    def __init__(self, data_dir):
        self.subdir = data_dir + "flickr_dataset/"
        self.caption_dir = self.subdir + "captions/"
        self.train_images_dir = self.subdir + "images/"
        self.val_images_dir = self.subdir + "images/"
        self.test_images_dir = self.subdir + "images/"
        self.captions_file = self.caption_dir + 'Flickr8k.token.txt'

    def get_name(self):
        return "flickr"

    def download_dataset(self):

        if not os.path.exists(self.caption_dir) or not os.path.exists(self.train_images_dir):
            print("DOWNLOADING FLICKR DATASET")
            shutil.rmtree(self.subdir, ignore_errors=True)

            os.system("git clone --progress -v https://github.com/luca-ant/WhatsSee_dataset.git " + self.subdir)
            # Repo.clone_from("https://github.com/luca-ant/WhatsSee_dataset.git", self.subdir, progress=Progress())

        else:
            print("Flickr dataset already exists")

    def load_captions(self):
        all_captions = collections.defaultdict(list)

        with open(self.captions_file, 'r') as f:

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

    def load_train_captions(self, num_train_examples):
        train_captions = collections.defaultdict(list)
        image_names = []
        with open(os.path.dirname(self.caption_dir) + "/Flickr_8k.trainImages.txt", 'r') as f:
            for line in f:
                image_names.append(line.strip())

        with open(self.captions_file, 'r') as f:

            for line in f:
                tokens = line.split()
                # take the first token as the image id, the rest as the description
                image_id, image_cap = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                image_cap = ' '.join(image_cap)
                if (image_id + ".jpg" in image_names):
                    train_captions[image_id].append(image_cap)

        if num_train_examples != 0:
            # Shuffle captions
            l = list(train_captions.items())
            random.shuffle(l)
            train_captions = dict(l)

            train_captions = dict(list(train_captions.items())[:num_train_examples])

        return train_captions

    def load_val_captions(self, num_val_examples):
        val_captions = collections.defaultdict(list)
        image_names = []
        with open(os.path.dirname(self.caption_dir) + "/Flickr_8k.devImages.txt", 'r') as f:
            for line in f:
                image_names.append(line.strip())

        with open(self.captions_file, 'r') as f:

            for line in f:
                tokens = line.split()
                # take the first token as the image id, the rest as the description
                image_id, image_cap = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                image_cap = ' '.join(image_cap)
                if (image_id + ".jpg" in image_names):
                    val_captions[image_id].append(image_cap)

        if num_val_examples != 0:
            # Shuffle captions
            l = list(val_captions.items())
            random.shuffle(l)
            val_captions = dict(l)

            val_captions = dict(list(val_captions.items())[:num_val_examples])

        return val_captions

    def load_test_captions(self, num_test_examples):
        test_captions = collections.defaultdict(list)
        image_names = []
        with open(os.path.dirname(self.caption_dir) + "/Flickr_8k.testImages.txt", 'r') as f:
            for line in f:
                image_names.append(line.strip())

        with open(self.captions_file, 'r') as f:

            for line in f:
                tokens = line.split()
                # take the first token as the image id, the rest as the description
                image_id, image_cap = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                image_cap = ' '.join(image_cap)
                if (image_id + ".jpg" in image_names):
                    test_captions[image_id].append(image_cap)

        if num_test_examples != 0:
            # Shuffle captions
            l = list(test_captions.items())
            random.shuffle(l)
            test_captions = dict(l)

            test_captions = dict(list(test_captions.items())[:num_test_examples])

        return test_captions

    def get_captions_of(self, image_name):
        original_captions = []
        with open(self.captions_file, 'r') as f:

            for line in f:
                tokens = line.split()
                # take the first token as the image id, the rest as the description
                image_id, image_cap = tokens[0], tokens[1:]
                image_id = image_id.split('.')[0]
                image_cap = ' '.join(image_cap)
                if (image_id + ".jpg" == image_name):
                    from process_data import clean_caption
                    original_captions.append(clean_caption(image_cap))
        return original_captions

    def get_test_image_names(self):
        image_names = []
        with open(os.path.dirname(self.caption_dir) + "/Flickr_8k.testImages.txt", 'r') as f:
            for line in f:
                image_names.append(line.strip())
        return image_names

    def load_images_name(self, images_id_list):
        images_name_list = []
        for id in images_id_list:
            image_name = id + '.jpg'
            if os.path.isfile(self.train_images_dir + image_name) or os.path.isfile(self.val_images_dir + image_name):
                images_name_list.append(image_name)

        return images_name_list

    def get_image_name(self, image_id):
        image_name = image_id + ".jpg"
        return image_name


class COCODataset():

    def __init__(self, data_dir):

        self.subdir = data_dir + "coco_dataset/"
        self.caption_dir = self.subdir + "captions/"
        self.train_images_dir = self.subdir + "images/train2017/"
        self.val_images_dir = self.subdir + "images/val2017/"
        self.test_images_dir = self.subdir + "images/test2017/"
        self.train_captions_file = self.caption_dir + 'annotations/captions_train2017.json'
        self.val_captions_file = self.caption_dir + 'annotations/captions_val2017.json'

    def get_name(self):
        return "coco"

    def download_dataset(self):

        name_of_zip = 'captions.zip'
        if not os.path.exists(self.caption_dir):
            os.makedirs(self.caption_dir, exist_ok=True)

            url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
            print("DOWNLOADING CAPTIONS FROM COCO DATASET")
            captions_zip = wget.download(url, self.subdir + name_of_zip)

            print("\nEXTRACTING ZIP CAPIOTN FILE IN " + self.caption_dir)

            with zipfile.ZipFile(captions_zip, 'r') as zip:
                zip.extractall(self.caption_dir)

            os.remove(captions_zip)

        else:
            print("COCO captions dataset already exists")

        name_of_zip = 'train2017.zip'

        if not os.path.exists(self.train_images_dir):

            os.makedirs(self.train_images_dir, exist_ok=True)

            url = 'http://images.cocodataset.org/zips/train2017.zip'
            print("DOWNLOADING TRAIN IMAGES FROM COCO DATASET")
            images_zip = wget.download(url, self.subdir + name_of_zip)

            print("\nEXTRACTING ZIP IMAGES FILE IN " + self.train_images_dir)

            with zipfile.ZipFile(images_zip, 'r') as zip:
                #                zip.extractall(self.train_images_dir)
                zip.extractall(self.subdir + "images/")

            os.remove(images_zip)


        else:
            print("COCO train images dataset already exists")

        name_of_zip = 'val2017.zip'

        if not os.path.exists(self.val_images_dir):

            os.makedirs(self.val_images_dir, exist_ok=True)

            url = 'http://images.cocodataset.org/zips/val2017.zip'
            print("DOWNLOADING VAL IMAGES FROM COCO DATASET")
            images_zip = wget.download(url, self.subdir + name_of_zip)

            print("\nEXTRACTING ZIP IMAGES FILE IN " + self.val_images_dir)

            with zipfile.ZipFile(images_zip, 'r') as zip:
                #                zip.extractall(self.val_images_dir)
                zip.extractall(self.subdir + "images/")

            os.remove(images_zip)


        else:
            print("COCO val images dataset already exists")

        name_of_zip = 'test2017.zip'

        if not os.path.exists(self.test_images_dir):

            os.makedirs(self.test_images_dir, exist_ok=True)

            url = 'http://images.cocodataset.org/zips/test2017.zip'
            print("DOWNLOADING TEST IMAGES FROM COCO DATASET")
            images_zip = wget.download(url, self.subdir + name_of_zip)

            print("\nEXTRACTING ZIP IMAGES FILE IN " + self.test_images_dir)

            with zipfile.ZipFile(images_zip, 'r') as zip:
                #                zip.extractall(self.test_images_dir)
                zip.extractall(self.subdir + "images/")

            os.remove(images_zip)


        else:
            print("COCO test images dataset already exists")

    def load_train_captions(self, num_train_examples):
        train_captions = collections.defaultdict(list)
        with open(self.train_captions_file, 'r') as f:
            captions = json.load(f)

        for c in captions['annotations']:
            caption_string = c['caption']
            image_id = c['image_id']
            image_id = "%012d" % (image_id)
            train_captions[image_id].append(caption_string)

        if num_train_examples != 0:
            # Shuffle captions
            l = list(train_captions.items())
            random.shuffle(l)
            train_captions = dict(l)

            train_captions = dict(list(train_captions.items())[:num_train_examples])

        return train_captions

    def load_val_captions(self, num_val_examples):
        val_captions = collections.defaultdict(list)
        with open(self.val_captions_file, 'r') as f:
            captions = json.load(f)

        for c in captions['annotations']:
            caption_string = c['caption']
            image_id = c['image_id']
            image_id = "%012d" % (image_id)
            val_captions[image_id].append(caption_string)

        if num_val_examples != 0:
            # Shuffle captions
            l = list(val_captions.items())
            random.shuffle(l)
            val_captions = dict(l)

            val_captions = dict(list(val_captions.items())[:num_val_examples])

        return val_captions

    def get_captions_of(self, image_name):
        original_captions = []
        if not original_captions:
            with open(self.train_captions_file, 'r') as f:
                captions = json.load(f)

            for c in captions['annotations']:
                caption_string = c['caption']
                image_id = c['image_id']
                image_id = "%012d" % (image_id)

                if (self.get_image_name(image_id) == image_name):
                    from process_data import clean_caption
                    original_captions.append(clean_caption(caption_string))

        if not original_captions:
            with open(self.val_captions_file, 'r') as f:
                captions = json.load(f)

            for c in captions['annotations']:
                caption_string = c['caption']
                image_id = c['image_id']
                image_id = "%012d" % (image_id)

                if (self.get_image_name(image_id) == image_name):
                    original_captions.append(clean_caption(caption_string))

        return original_captions

    def get_test_image_names(self):
        return os.listdir(self.test_images_dir)

    def load_images_name(self, images_id_list):
        images_name_list = []
        for id in images_id_list:
            image_name = id + ".jpg"
            if os.path.isfile(self.train_images_dir + image_name) or os.path.isfile(self.val_images_dir + image_name):
                images_name_list.append(image_name)
        return images_name_list

    def get_image_name(self, image_id):
        image_name = image_id + ".jpg"
        return image_name
