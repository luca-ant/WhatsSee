import collections
import json
import os
import tensorflow as tf


class Dataset:
    @staticmethod
    def download_dataset(dataset):
        return dataset.download_dataset()

    @staticmethod
    def load_captions(dataset):
        return dataset.load_captions()

    @staticmethod
    def load_images_name(dataset, images_id_list):
        return dataset.load_images_name(images_id_list)


class FlickrDataset():
    caption_file_path = ""
    dataset_dir_path = ""

    def __init__(self):
        super().__init__()


class COCODataset():
    caption_file_path = ""
    dataset_dir_path = ""

    def __init__(self):
        super().__init__()

    def download_dataset(self):
        data_path = "./data/"

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
        #   name_of_zip = 'test2014.zip'

        if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):

            image_zip = tf.keras.utils.get_file(name_of_zip,
                                                cache_subdir=os.path.abspath("."),
                                                origin='http://images.cocodataset.org/zips/train2014.zip',
                                                extract=True)
            dataset_dir_path = os.path.dirname(image_zip) + '/train2014/'
            """

            image_zip = tf.keras.utils.get_file(name_of_zip,
                                                cache_subdir=os.path.abspath("."),
                                                origin='http://images.cocodataset.org/zips/test2014.zip',
                                                extract=True)
            dataset_dir_path = os.path.dirname(image_zip) + '/test2014/'
            """

        else:
            dataset_dir_path = os.path.abspath('.') + '/train2014/'
            #    dataset_dir_path = os.path.abspath('.') + '/test2014/'
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

    def load_images_name(self, images_id_list):
        images_name_list = []
        for id in images_id_list:
            image_name = 'COCO_train2014_' + '%012d.jpg' % (id)
            if os.path.isfile(self.dataset_dir_path + "/" + image_name):
                images_name_list.append(image_name)

        return images_name_list
