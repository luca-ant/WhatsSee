#!/usr/bin/python
import os
import sys
import logging
import traceback

from keras.engine.saving import load_model, save_model

from model import Dataset, create_NN
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, \
    predict_caption, store_vocabulary, load_vocabulary, data_generator, clean_captions, load_train_data, store_train_data, store_val_data, load_val_data
from keras.callbacks import ModelCheckpoint, Callback

import tensorflow as tf

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

    def __init__(self, working_dir):
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

    def resume(self):
        if os.path.isdir(self.train_dir):
            print("RESUME LAST TRAINING")

            # load dataset name
            with open(self.dataset_name_file, "r") as f:
                dataset_name = f.readline().strip()
            dataset = Dataset.create_dataset(dataset_name, self.data_dir)

            # load last epoch number
            with open(self.epoch_file, "r") as f:
                last_epoch = int(f.readline().strip())

            # load vocabulary, train and val data
            vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(self.vocabulary_dir)
            model = load_model(self.model_file)
            train_captions, train_images_as_vector = load_train_data(self.train_dir)
            val_captions, val_images_as_vector = load_val_data(self.train_dir)

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            if not os.path.isdir(self.weights_dir):
                os.makedirs(self.weights_dir)

            # callbacks
            save_weights_callback = ModelCheckpoint(self.weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)
            save_epoch_callback = EpochSaver(last_epoch + 1, self.epoch_file)
            save_model_callback = ModelCheckpoint(self.model_file, verbose=1, period=1)

            # params
            batch_size = 16
            steps = (len(train_captions) // batch_size) + 1
            model.summary()

            # prepare train and val data
            x_val_text, x_val_image, y_val_caption = prepare_data(dataset, val_captions, val_images_as_vector, word_index_dict, len(vocabulary), max_cap_len)
            generator = data_generator(dataset, train_captions, train_images_as_vector, word_index_dict, max_cap_len,
                                       len(vocabulary), batch_size)

            print("TRAINING MODEL")
            history = model.fit_generator(generator, epochs=150, steps_per_epoch=steps, verbose=2, validation_data=([x_val_image, x_val_text], y_val_caption),
                                          callbacks=[save_weights_callback, save_model_callback, save_epoch_callback], initial_epoch=last_epoch)

            loss = history.history['loss'][-1]
            acc = history.history['acc'][-1]

            print("SAVING WEIGHTS TO " + self.weights_file)

            model.save_weights(self.weights_file, True)
            print("TRAINING COMPLETE!")
            print("LOSS: {:5.2f}".format(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))

            return history

        else:
            print("LAST TRAINING DATA NOT FOUND")
            exit(3)

    def train(self, dataset, num_train_examples, num_val_examples):
        print("START NEW TRAINING")
        1
        if os.path.isdir(self.train_dir):
            os.system("rm -rf " + self.train_dir)

        captions_file_path, images_dir_path = Dataset.download_dataset(dataset)

        # load captions from dataset
        train_captions = Dataset.load_train_captions(dataset, num_train_examples)
        train_captions = clean_captions(train_captions)
        train_images_name_list = Dataset.load_images_name(dataset, train_captions.keys())
        train_captions = add_start_end_token(train_captions)
        train_captions_list = to_captions_list(train_captions)

        val_captions = Dataset.load_val_captions(dataset, num_val_examples)
        val_captions = clean_captions(val_captions)
        val_images_name_list = Dataset.load_images_name(dataset, val_captions.keys())
        val_captions = add_start_end_token(val_captions)

        # generate vocabulary
        max_cap_len = max(len(d.split()) for d in train_captions_list)
        vocabulary = generate_vocabulary(train_captions_list)
        print("VOCABULARY SIZE: " + str(len(vocabulary)))
        print("MAX CAPTION LENGTH: " + str(max_cap_len))
        vocabulary.append("0")
        index_word_dict = {}
        word_index_dict = {}
        i = 1
        for w in vocabulary:
            word_index_dict[w] = i
            index_word_dict[i] = w
            i += 1

        model = create_NN(len(vocabulary), max_cap_len)

        # load images from dataset
        train_images_as_vector = preprocess_images(images_dir_path, train_images_name_list)
        val_images_as_vector = preprocess_images(images_dir_path, val_images_name_list)

        # store vocabulary, train and val data
        store_vocabulary(self.vocabulary_dir, vocabulary, word_index_dict, index_word_dict, max_cap_len)
        store_train_data(self.train_dir, train_captions, train_images_as_vector)
        store_val_data(self.train_dir, val_captions, val_images_as_vector)
        save_model(model, self.model_file)
        with open(self.dataset_name_file, "w") as f:
            f.write(dataset.get_name())

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)

        # callbacks
        save_weights_callback = ModelCheckpoint(self.weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)
        save_epoch_callback = EpochSaver(1, self.epoch_file)
        save_model_callback = ModelCheckpoint(self.model_file, verbose=1, period=1)

        batch_size = 16
        steps = (len(train_captions) // batch_size) + 1

        model.summary()

        # prepare train and val data
        x_val_text, x_val_image, y_val_caption = prepare_data(dataset, val_captions, val_images_as_vector, word_index_dict, len(vocabulary), max_cap_len)
        generator = data_generator(dataset, train_captions, train_images_as_vector, word_index_dict, max_cap_len,
                                   len(vocabulary), batch_size)

        print("TRAINING MODEL")
        history = model.fit_generator(generator, epochs=150, steps_per_epoch=steps, verbose=2, validation_data=([x_val_image, x_val_text], y_val_caption),
                                      callbacks=[save_weights_callback, save_model_callback, save_epoch_callback])

        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]

        print("SAVING WEIGHTS TO " + self.weights_file)

        model.save_weights(self.weights_file, True)
        print("TRAINING COMPLETE!")
        print("LOSS: {:5.2f}".format(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))

        return history

    def predict(self, image_name):
        vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(self.vocabulary_dir)

        print("VOCABULARY SIZE: " + str(len(vocabulary)))
        print("MAX CAPTION LENGTH: " + str(max_cap_len))

        model = create_NN(len(vocabulary), max_cap_len)

        model.load_weights(self.weights_file)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        predicted_caption = predict_caption(model, image_name, max_cap_len,
                                            word_index_dict, index_word_dict)

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
    num_train_examples = 0
    num_val_examples = 0
    image_file_name = ""

    # read args
    if mode == "train":
        num_args = 2 + (2 * 3)
        num_train_examples = 6000
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
    ws = WhatsSee(working_dir)
    dataset = Dataset.create_dataset(dataset_name, ws.data_dir)

    # select mode
    if mode == "train":

        hystory = ws.train(dataset, num_train_examples, num_val_examples)

    elif mode == "resume":

        hystory = ws.resume()

    elif mode == "predict":

        predicted_caption = ws.predict(image_file_name)

    exit(0)
