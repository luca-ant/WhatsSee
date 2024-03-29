#!/usr/bin/python
import os
import shutil
import sys
import logging
import traceback
import warnings
import nltk
import numpy as np
import progressbar
from PIL import Image
from keras.optimizers import Adam
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from dataset import Dataset
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, store_vocabulary, load_vocabulary, data_generator, \
    clean_captions, load_train_data, store_train_data, store_val_data, load_val_data
from keras.engine.saving import load_model, save_model
from keras.callbacks import ModelCheckpoint, Callback
from keras import models, metrics
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras import Input, Model
from keras.layers import Dropout, Dense, LSTM, Embedding, add

import tensorflow as tf
from tensorflow.python.keras.preprocessing import image

warnings.simplefilter("ignore")

tf.get_logger().setLevel(logging.ERROR)


class EpochSaver(Callback):
    def __init__(self, start_epoch, last_epoch_file):
        self.epoch = start_epoch
        self.last_epoch_file = last_epoch_file

    def on_epoch_end(self, epoch, logs={}):
        with open(self.last_epoch_file, "w") as f:
            f.write(str(self.epoch))
        self.epoch += 1


def create_NN(vocab_size, max_cap_len):
    print("CREATING MODEL")
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
            self.generated_captions_dir = working_dir + "/output/generated_captions/"
            self.captioned_images_dir = working_dir + "/output/captioned_images/"
            self.vocabulary_dir = self.data_dir + "vocabulary/"
            self.weights_dir = self.data_dir + "weights/"
            self.train_dir = self.data_dir + "training/"

            self.weights_file = self.weights_dir + "weights.h5"
            self.model_file = self.train_dir + "model.h5"
            self.dataset_name_file = self.train_dir + "dataset_name.txt"
            self.last_epoch_file = self.train_dir + "last_epoch.txt"
            self.total_epoch_file = self.train_dir + "total_epoch.txt"

            self.dataset = Dataset.create_dataset(dataset_name, self.data_dir)
            self.last_epoch = 0
            self.batch_size = 16
            self.total_epochs = 50

            self.model = None
            self.modelvgg = None
            self.train_captions = None
            self.val_captions = None
            self.test_captions = None
            self.train_images_as_vector = None
            self.val_images_as_vector = None
            self.vocabulary = None
            self.word_index_dict = None
            self.index_word_dict = None
            self.max_cap_len = None

            if not os.path.isdir(self.data_dir):
                os.makedirs(self.data_dir)

            if not os.path.isdir(self.captioned_images_dir):
                os.makedirs(self.captioned_images_dir)

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
        self.last_epoch_file = self.train_dir + "last_epoch.txt"

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
    """

    def set_dataset(self, dataset_name):
        self.dataset = Dataset.create_dataset(dataset_name, self.data_dir)

    def set_total_epochs(self, total_epochs):
        self.total_epochs = total_epochs

    def download_dataset(self):
        self.dataset.download_dataset()

    def process_raw_data(self, num_train_examples, num_val_examples):

        # load captions from dataset
        train_captions = self.dataset.load_train_captions(num_train_examples)
        train_captions = clean_captions(train_captions)
        train_images_name_list = self.dataset.load_images_name(train_captions.keys())
        self.train_captions = add_start_end_token(train_captions)
        train_captions_list = to_captions_list(self.train_captions)

        val_captions = self.dataset.load_val_captions(num_val_examples)
        val_captions = clean_captions(val_captions)
        val_images_name_list = self.dataset.load_images_name(val_captions.keys())
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

        # load images from dataset
        self.train_images_as_vector = preprocess_images(self.dataset.train_images_dir, train_images_name_list)
        self.val_images_as_vector = preprocess_images(self.dataset.val_images_dir, val_images_name_list)

        # create model
        self.model = create_NN(len(self.vocabulary), self.max_cap_len)

        return

    def save_data_on_disk(self):
        # store vocabulary, train and val data
        store_vocabulary(self.vocabulary_dir, self.vocabulary, self.word_index_dict, self.index_word_dict, self.max_cap_len)
        store_train_data(self.train_dir, self.train_captions, self.train_images_as_vector)
        store_val_data(self.train_dir, self.val_captions, self.val_images_as_vector)
        save_model(self.model, self.model_file)
        with open(self.dataset_name_file, "w") as f:
            f.write(self.dataset.get_name())
        with open(self.last_epoch_file, "w") as f:
            f.write(str(self.last_epoch))
        with open(self.total_epoch_file, "w") as f:
            f.write(str(self.total_epochs))
        return

    def load_data_from_disk(self):

        # load dataset name
        with open(self.dataset_name_file, "r") as f:
            dataset_name = f.readline().strip()
        self.dataset = Dataset.create_dataset(dataset_name, self.data_dir)

        # load last epoch number
        with open(self.last_epoch_file, "r") as f:
            self.last_epoch = int(f.readline().strip())
        print("RESUME FROM " + str(self.last_epoch) + " EPOCH")
        # load total epoch number
        with open(self.total_epoch_file, "r") as f:
            self.total_epochs = int(f.readline().strip())

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
        opt = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if self.last_epoch >= self.total_epochs:
            print("LAST EPOCH TOO MUCH BIG")
            return

        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)

        # callbacks
        save_weights_callback = ModelCheckpoint(self.weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)
        save_epoch_callback = EpochSaver(self.last_epoch + 1, self.last_epoch_file)
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
        history = self.model.fit_generator(train_data_generator, epochs=self.total_epochs, steps_per_epoch=steps_train, verbose=2, validation_data=val_data_generator,
                                           validation_steps=steps_val, callbacks=[save_weights_callback, save_model_callback, save_epoch_callback],
                                           initial_epoch=self.last_epoch)

        # for i in range(self.last_epoch, self.total_epochs):
        #     print("EPOCH: " + str(i + 1) + "/" + str(self.total_epochs))
        #     history = self.model.fit_generator(train_data_generator, epochs=1, steps_per_epoch=steps_train, verbose=2, validation_data=val_data_generator,
        #                                        validation_steps=steps_val, use_multiprocessing=True,
        #                                        callbacks=[save_weights_callback, save_model_callback, save_epoch_callback])

        print("SAVING WEIGHTS TO " + self.weights_file)

        self.model.save_weights(self.weights_file, True)
        print("TRAINING COMPLETE!")

        if os.path.isdir(self.train_dir):
            shutil.rmtree(self.train_dir, ignore_errors=True)

        loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        acc = history.history['acc'][-1]
        val_acc = history.history['val_acc'][-1]
        print(
            "LOSS: {:5.2f}".format(loss) + " - ACC: {:5.2f}%".format(100 * acc) + " - VAL_LOSS: {:5.2f}".format(val_loss) + " - VAL_ACC: {:5.2f}%".format(100 * val_acc))
        return history

    def restore_nn(self):
        self.vocabulary, self.word_index_dict, self.index_word_dict, self.max_cap_len = load_vocabulary(self.vocabulary_dir)

        print("VOCABULARY SIZE: " + str(len(self.vocabulary)))
        print("MAX CAPTION LENGTH: " + str(self.max_cap_len))

        self.model = create_NN(len(self.vocabulary), self.max_cap_len)

        self.model.load_weights(self.weights_file)

        #        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def predict_caption(self, image_name):

        if self.modelvgg == None:
            self.modelvgg = VGG16(weights="imagenet")
            self.modelvgg.layers.pop()
            self.modelvgg = models.Model(inputs=self.modelvgg.inputs, outputs=self.modelvgg.layers[-1].output)

        img = image.load_img(image_name, target_size=(224, 224, 3))

        img = image.img_to_array(img)

        img = preprocess_input(img)
        img = self.modelvgg.predict(img.reshape((1,) + img.shape[:3]))

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

    ################################################

    def resume(self):
        if os.path.isdir(self.train_dir):
            print("RESUMING LAST TRAINING")
            self.load_data_from_disk()
            self.start_train()
        else:
            print("LAST TRAINING DATA NOT FOUND")

    def train(self, num_train_examples, num_val_examples):
        print("STARTING NEW TRAINING")

        self.clean_last_training_data()
        self.download_dataset()
        self.process_raw_data(num_train_examples, num_val_examples)
        self.save_data_on_disk()
        self.start_train()

    def evaluate_model(self, num_test_examples):

        if self.model == None:
            self.restore_nn()

        self.set_dataset("flickr")
        self.download_dataset()

        test_captions = self.dataset.load_test_captions(num_test_examples)
        test_captions = clean_captions(test_captions)
        test_captions = add_start_end_token(test_captions)
        originals, predicted = list(), list()

        print("EVALUATING MODEL")

        with progressbar.ProgressBar(max_value=len(test_captions.keys())) as bar:
            i = 0
            out = []

            for key, desc_list in test_captions.items():
                i += 1
                caption = self.predict_caption(self.dataset.train_images_dir + self.dataset.get_image_name(key))

                references = [d.split()[1:-1] for d in desc_list]
                originals.append(references)
                predicted.append(caption.split())
                bar.update(i)

                ###########

                for c in desc_list:
                    # reference.append(c.split())

                    score = sentence_bleu([c.strip().split()[1:-1]], caption.strip().split(), weights=(1.0, 0, 0, 0))
                    if score > 0.4:
                        out.append(key + ".jpg\n" + caption + "\n" + c + " (BLEU SCORE: {:4.1f}%)".format(100 * score)+"\n")

                ###########

        for s in out:
            print(s)

        evaluation_as_strings = []

        # calculate BLEU score
        evaluation_as_strings.append('BLEU-1: {:4.1f}%'.format(corpus_bleu(originals, predicted, weights=(1.0, 0, 0, 0)) * 100))
        evaluation_as_strings.append('BLEU-2: {:4.1f}%'.format(corpus_bleu(originals, predicted, weights=(0.5, 0.5, 0, 0)) * 100))
        evaluation_as_strings.append('BLEU-3: {:4.1f}%'.format(corpus_bleu(originals, predicted, weights=(0.3, 0.3, 0.3, 0)) * 100))
        evaluation_as_strings.append('BLEU-4: {:4.1f}%'.format(corpus_bleu(originals, predicted, weights=(0.25, 0.25, 0.25, 0.25)) * 100))

        for e in evaluation_as_strings:
            print(e)

        return evaluation_as_strings

    def test(self, image_name):
        print("TESTING")

        self.download_dataset()

        caption = self.predict(image_name)

        original_captions = self.dataset.get_captions_of(os.path.basename(image_name))

        if original_captions:
            reference = []
            bleu_scores = []

            for c in original_captions:
                # reference.append(c.split())
                bleu_scores.append(sentence_bleu([c.strip().split()], caption.strip().split(), weights=(1.0, 0, 0, 0)))

            bleu_scores_string = []
            for b in bleu_scores:
                bleu_scores_string.append("BLEU SCORE: {:4.1f}%".format(100 * b))
            #
            # for c, b in zip(original_captions, bleu_scores):
            #     print(c + " (" + str(b) + ")")

            for c, b in zip(original_captions, bleu_scores_string):
                print(c + " (" + b + ")")

    def predict(self, image_name):
        print("GENERATING CAPTION")

        if self.model == None:
            self.restore_nn()

        predicted_caption = self.predict_caption(image_name)

        if not os.path.isdir(self.generated_captions_dir):
            os.makedirs(self.generated_captions_dir)

        if not os.path.isdir(self.captioned_images_dir):
            os.makedirs(self.captioned_images_dir)

        file_name = os.path.basename(image_name)
        if not os.path.isfile(self.captioned_images_dir + file_name):
            shutil.copyfile(image_name, self.captioned_images_dir + file_name)

        caption_file_name = "".join(file_name.split(".")[:-1]) + ".txt"

        with open(self.generated_captions_dir + caption_file_name, "w") as f:
            f.write(predicted_caption)

        print("GENERATED CAPTION: " + predicted_caption)
        return predicted_caption


def usage():
    print("Usage: " + sys.argv[0] + " [train | generate | resume | test | evaluate] ")
    exit(1)


def usage_train():
    print("Usage: " + sys.argv[0] + " train -d [coco | flickr] -nt NUMBER -nv NUMBER [-ne NUMBER]")
    exit(2)


def usage_generate():
    print("Usage: " + sys.argv[0] + " generate -f YOUR_IMAGE_FILE")
    exit(2)


def usage_test():
    print("Usage: " + sys.argv[0] + " test -f TEST_IMAGE_FILE")
    exit(2)


def usage_resume():
    print("Usage: " + sys.argv[0] + " resume")
    exit(2)


def usage_evaluate():
    print("Usage: " + sys.argv[0] + " evaluate -n NUMBER")
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
    num_test_examples = 0
    total_epochs = 50
    image_file_name = ""

    # read args
    if mode == "train":
        num_args = 2 + (2 * 4)
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

            elif op == "-ne":
                try:
                    total_epochs = int(val)

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


    elif mode == "evaluate":
        num_args = 2 + (2 * 1)
        num_args = min(num_args, len(sys.argv))

        # if len(sys.argv) < num_args:
        #     usage_evaluate()

        for i in range(2, num_args, 2):
            op = sys.argv[i]
            val = sys.argv[i + 1]

            if op == "-n":
                try:
                    num_test_examples = int(val)

                except:
                    print("Invalid value's option: " + val)
                    usage_evaluate()
            else:
                print("Invalid option: " + op)

                usage_evaluate()

    elif mode == "generate":
        num_args = 2 + (2 * 1)

        if len(sys.argv) < num_args:
            usage_generate()

        for i in range(2, num_args, 2):
            op = sys.argv[i]
            val = sys.argv[i + 1]

            if op == "-f":
                image_file_name = val
                if not os.path.isfile(image_file_name):
                    print("404 File Not Found: " + image_file_name)
                    usage_generate()
            else:
                print("Invalid option: " + op)

                usage_generate()


    elif mode == "test":
        num_args = 2 + (2 * 1)

        if len(sys.argv) < num_args:
            usage_test()

        for i in range(2, num_args, 2):
            op = sys.argv[i]
            val = sys.argv[i + 1]

            if op == "-f":
                image_file_name = val
                if not os.path.isfile(image_file_name):
                    print("404 File Not Found: " + image_file_name)
                    usage_test()
            else:
                print("Invalid option: " + op)

                usage_test()

    # create objects
    working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    ws = WhatsSee(dataset_name, working_dir)
    ws.set_total_epochs(total_epochs)

    # select mode
    if mode == "train":

        ws.train(num_train_examples, num_val_examples)

    elif mode == "resume":

        ws.resume()

    elif mode == "generate":

        predicted_caption = ws.predict(image_file_name)
        im = Image.open(image_file_name)
        im.show()

    elif mode == "test":
        ws.test(image_file_name)
        im = Image.open(image_file_name)
        im.show()

    elif mode == "evaluate":
        ws.evaluate_model(num_test_examples)
