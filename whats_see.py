#!/usr/bin/python
import os
import shutil
import sys
import logging
import traceback
import numpy as np

from model import Dataset, create_NN
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, store_vocabulary, load_vocabulary, data_generator, \
    clean_captions, load_train_data, store_train_data, store_val_data, load_val_data
from keras.engine.saving import load_model, save_model
from keras.callbacks import ModelCheckpoint, Callback
from keras import models
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image

tf.get_logger().setLevel(logging.ERROR)


class EpochSaver(Callback):
    def __init__(self, start_epoch, epoch_file):
        self.epoch = start_epoch
        self.epoch_file = epoch_file

    def on_epoch_end(self, epoch, logs={}):
        with open(self.epoch_file, "w") as f:
            f.write(str(self.epoch))
        self.epoch += 1


class WhatsSee():
    __instance = None

    @staticmethod
    def get_instance():

        if WhatsSee.__instance == None:
            WhatsSee(".")
        return WhatsSee.__instance

    def __init__(self, dataset_name, working_dir):
        # private constructor
        if WhatsSee.__instance != None:
            raise Exception("WhatsSee class is a singleton! Use WhatsSee.get_instance()")
        else:

            self.data_dir = working_dir + "/data/"
            self.vocabulary_dir = self.data_dir + "vocabulary/"
            self.weights_dir = self.data_dir + "weights/"
            self.train_dir = self.data_dir + "train/"

            self.weights_file = self.weights_dir + "weights.h5"
            self.model_file = self.train_dir + "model.h5"
            self.dataset_name_file = self.train_dir + "dataset_name.txt"
            self.epoch_file = self.train_dir + "last_epoch.txt"

            self.dataset = Dataset.create_dataset(dataset_name, self.data_dir)
            self.last_epoch = 0
            self.batch_size = 16

            self.model = None
            self.train_captions = None
            self.val_captions = None
            self.train_images_as_vector = None
            self.val_images_as_vector = None
            self.vocabulary = None
            self.word_index_dict = None
            self.index_word_dict = None
            self.max_cap_len = None

            if not os.path.isdir(self.data_dir):
                os.makedirs(self.data_dir)

            WhatsSee.__instance = self

    """
    def __init__(self, working_dir):

        self.data_dir = working_dir + "/data/"
        self.vocabulary_dir = self.data_dir + "vocabulary/"
        self.weights_dir = self.data_dir + "weights/"
        self.train_dir = self.data_dir + "train/"

        self.weights_file = self.weights_dir + "weights.h5"
        self.model_file = self.train_dir + "model.h5"
        self.dataset_name_file = self.train_dir + "dataset_name.txt"
        self.epoch_file = self.train_dir + "last_epoch.txt"

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
    """

    def process_raw_data(self, num_train_examples, num_val_examples):

        captions_file_path, images_dir_path = Dataset.download_dataset(self.dataset)

        # load captions from dataset
        train_captions = Dataset.load_train_captions(self.dataset, num_train_examples)
        train_captions = clean_captions(train_captions)
        train_images_name_list = Dataset.load_images_name(self.dataset, train_captions.keys())
        self.train_captions = add_start_end_token(train_captions)
        train_captions_list = to_captions_list(train_captions)

        val_captions = Dataset.load_val_captions(self.dataset, num_val_examples)
        val_captions = clean_captions(val_captions)
        val_images_name_list = Dataset.load_images_name(self.dataset, val_captions.keys())
        self.val_captions = add_start_end_token(val_captions)

        # generate vocabulary
        self.max_cap_len = max(len(d.split()) for d in train_captions_list)
        self.vocabulary = generate_vocabulary(train_captions_list)
        print("VOCABULARY SIZE: " + str(len(self.vocabulary)))
        print("MAX CAPTION LENGTH: " + str(self.max_cap_len))
        self.vocabulary.append("0")
        self.index_word_dict = {}
        self.word_index_dict = {}
        i = 1
        for w in self.vocabulary:
            self.word_index_dict[w] = i
            self.index_word_dict[i] = w
            i += 1

        self.model = create_NN(len(self.vocabulary), self.max_cap_len)

        # load images from dataset
        self.train_images_as_vector = preprocess_images(images_dir_path, train_images_name_list)
        self.val_images_as_vector = preprocess_images(images_dir_path, val_images_name_list)

        return

    def save_data_on_disk(self):
        # store vocabulary, train and val data
        store_vocabulary(self.vocabulary_dir, self.vocabulary, self.word_index_dict, self.index_word_dict, self.max_cap_len)
        store_train_data(self.train_dir, self.train_captions, self.train_images_as_vector)
        store_val_data(self.train_dir, self.val_captions, self.val_images_as_vector)
        save_model(self.model, self.model_file)
        with open(self.dataset_name_file, "w") as f:
            f.write(self.dataset.get_name())
        with open(self.epoch_file, "w") as f:
            f.write(str(self.last_epoch))

        return

    def load_data_from_disk(self):

        # load dataset name
        with open(self.dataset_name_file, "r") as f:
            dataset_name = f.readline().strip()
        self.dataset = Dataset.create_dataset(dataset_name, self.data_dir)

        # load last epoch number
        with open(self.epoch_file, "r") as f:
            self.last_epoch = int(f.readline().strip())

        # load vocabulary, train and val data
        self.vocabulary, self.word_index_dict, self.index_word_dict, self.max_cap_len = load_vocabulary(self.vocabulary_dir)
        self.model = load_model(self.model_file)
        self.train_captions, self.train_images_as_vector = load_train_data(self.train_dir)
        self.val_captions, self.val_images_as_vector = load_val_data(self.train_dir)

        return

    def clean_last_training_data(self):
        if os.path.isdir(self.train_dir):
            # os.system("rm -rf " + self.train_dir)
            shutil.rmtree(self.train_dir, ignore_errors=True)

    def start_train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)

        # callbacks
        save_weights_callback = ModelCheckpoint(self.weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)
        save_epoch_callback = EpochSaver(self.last_epoch + 1, self.epoch_file)
        save_model_callback = ModelCheckpoint(self.model_file, verbose=1, period=1)

        # params
        steps_train = (len(self.train_captions) // self.batch_size) + 1
        steps_val = (len(self.val_captions) // self.batch_size) + 1
        self.model.summary()

        # prepare train and val data generator
        train_data_generator = data_generator(self.dataset, self.train_captions, self.train_images_as_vector, self.word_index_dict, self.max_cap_len,
                                              len(self.vocabulary), self.batch_size)
        val_data_generator = data_generator(self.dataset, self.val_captions, self.val_images_as_vector, self.word_index_dict, self.max_cap_len, len(self.vocabulary),
                                            self.batch_size)

        print("TRAINING MODEL")
        history = self.model.fit_generator(train_data_generator, epochs=300, steps_per_epoch=steps_train, verbose=2, validation_data=val_data_generator,
                                           validation_steps=steps_val, callbacks=[save_weights_callback, save_model_callback, save_epoch_callback],
                                           initial_epoch=self.last_epoch)

        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]

        print("SAVING WEIGHTS TO " + self.weights_file)

        self.model.save_weights(self.weights_file, True)
        print("TRAINING COMPLETE!")
        print("LOSS: {:5.2f}".format(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))

        return history

    def restore_nn(self):
        self.vocabulary, self.word_index_dict, self.index_word_dict, self.max_cap_len = load_vocabulary(self.vocabulary_dir)

        print("VOCABULARY SIZE: " + str(len(self.vocabulary)))
        print("MAX CAPTION LENGTH: " + str(self.max_cap_len))

        self.model = create_NN(len(self.vocabulary), self.max_cap_len)

        self.model.load_weights(self.weights_file)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def predict_caption(self, image_name):
        modelvgg = VGG16(include_top=True)

        modelvgg.layers.pop()
        modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)

        img = image.load_img(image_name, target_size=(224, 224, 3))

        img = image.img_to_array(img)

        img = preprocess_input(img)
        img = modelvgg.predict(img.reshape((1,) + img.shape[:3]))

        in_text = 'start_seq'
        for i in range(self.max_cap_len):
            sequence = [self.word_index_dict[w] for w in in_text.split() if w in self.word_index_dict]
            sequence = pad_sequences([sequence], maxlen=self.max_cap_len)
            yhat = self.model.predict([img, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.index_word_dict[yhat]
            in_text += ' ' + word
            if word == 'end_seq':
                break
        caption = in_text.split()
        caption = caption[1:-1]
        caption = ' '.join(caption)

        return caption

    def resume(self):
        if os.path.isdir(self.train_dir):
            print("RESUME LAST TRAINING")
            self.load_data_from_disk()
            self.start_train()
        else:
            print("LAST TRAINING DATA NOT FOUND")

    def train(self, num_train_examples, num_val_examples):
        print("START NEW TRAINING")

        self.clean_last_training_data()
        self.process_raw_data(num_train_examples, num_val_examples)
        self.save_data_on_disk()
        self.start_train()

    def predict(self, image_name):

        if self.model == None:
            self.restore_nn()

        predicted_caption = self.predict_caption(image_name)

        print(predicted_caption)

        return predicted_caption


def usage():
    print("Usage: " + sys.argv[0] + " [train | predict | resume] ")
    exit(1)


def usage_train():
    print("Usage: " + sys.argv[0] + " train -d [coco | flickr] -nt NUMBER -nv NUMBER")
    exit(2)


def usage_predict():
    print("Usage: " + sys.argv[0] + " predict -f YOUR_IMAGE_FILE")
    exit(2)


def usage_resume():
    print("Usage: " + sys.argv[0] + " resume")
    exit(2)


## START PROGRAM


if __name__ == "__main__":

    # check args
    if len(sys.argv) < 2:
        usage()
        exit(1)

    mode = sys.argv[1]

    # default values
    dataset_name = "flickr"
    num_train_examples = 6000
    num_val_examples = 1000
    image_file_name = ""

    # read args
    if mode == "train":
        num_args = 2 + (2 * 3)
        num_args = min(num_args, len(sys.argv))
        #        if len(sys.argv) < num_args:
        #            usage_train()

        for i in range(2, num_args, 2):
            op = sys.argv[i]
            val = sys.argv[i + 1]

            if op == "-d":
                if val == "coco" or val == "flickr":
                    dataset_name = val

                else:
                    print("Invalid value's option: " + val)
                    usage_train()

            elif op == "-nt":
                try:
                    num_train_examples = int(val)

                except:
                    print("Invalid value's option: " + val)
                    usage_train()

            elif op == "-nv":
                try:
                    num_val_examples = int(val)

                except:
                    print("Invalid value's option: " + val)
                    usage_train()
            else:
                print("Invalid option: " + op)
                usage_train()


    elif mode == "resume":
        num_args = 2 + (2 * 0)

        if len(sys.argv) != num_args:
            usage_resume()


    elif mode == "predict":
        num_args = 2 + (2 * 1)

        if len(sys.argv) < num_args:
            usage_predict()

        for i in range(2, num_args, 2):
            op = sys.argv[i]
            val = sys.argv[i + 1]

            if op == "-f":
                image_file_name = val
                if not os.path.isfile(image_file_name):
                    print("404 File Not Found: " + image_file_name)
                    usage_predict()
            else:
                print("Invalid option: " + op)

                usage_predict()

    # create objects
    working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    ws = WhatsSee(dataset_name, working_dir)

    # select mode
    if mode == "train":

        hystory = ws.train(num_train_examples, num_val_examples)

    elif mode == "resume":

        hystory = ws.resume()

    elif mode == "predict":

        predicted_caption = ws.predict(image_file_name)

    exit(0)
