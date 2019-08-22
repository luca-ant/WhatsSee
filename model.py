import collections
import json
import os
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
    def download_dataset(dataset):
        return dataset.download_dataset()

    @staticmethod
    def load_captions(dataset):
        return dataset.load_captions()

    @staticmethod
    def load_images_name(dataset, images_id_list):
        return dataset.load_images_name(images_id_list)

    @staticmethod
    def get_image_name(dataset, image_id):
        return dataset.get_image_name(image_id)


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

    def get_image_name(self, image_id):
        image_name = "COCO_train2014_"+ "%012d.jpg" % (image_id)
        return image_name
