#!/usr/bin/python

import os
import sys
import logging
import traceback

from keras.engine.saving import load_model, save_model

from model import COCODataset, FlickrDataset, Dataset, create_NN
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, \
    predict_caption, store_vocabulary, load_vocabulary, data_generator, clean_captions, load_train_data, store_train_data, store_val_data, load_val_data
from keras.callbacks import ModelCheckpoint, Callback

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

current_work_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

data_dir = current_work_dir + "/data/"
vocabulary_dir = data_dir + "vocabulary/"
weights_dir = data_dir + "weights/"
train_dir = data_dir + "train/"

weights_file = weights_dir + "weights.h5"
model_file = train_dir + "model.h5"
dataset_name_file = train_dir + "dataset_name.txt"
epoch_file = train_dir + "last_epoch.txt"


class EpochSaver(Callback):
    def __init__(self, start_epoch):
        self.epoch = start_epoch

    def on_epoch_end(self, epoch, logs={}):
        with open(epoch_file, "w") as f:
            f.write(str(self.epoch))
        self.epoch += 1


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


def resume():
    if os.path.isdir(train_dir):
        print("RESUME LAST TRAINING")

        # load dataset name
        with open(dataset_name_file, "r") as f:
            dataset_name = f.readline().strip()
        dataset = Dataset.create_dataset(dataset_name)

        # load last epoch number
        with open(epoch_file, "r") as f:
            last_epoch = int(f.readline().strip())

        # load vocabulary, train and val data
        vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(vocabulary_dir)
        model = load_model(model_file)
        train_captions, train_images_as_vector = load_train_data(train_dir)
        val_captions, val_images_as_vector = load_val_data(train_dir)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)

        # callbacks
        save_weights_callback = ModelCheckpoint(weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)
        save_epoch_callback = EpochSaver(last_epoch + 1)
        save_model_callback = ModelCheckpoint(model_file, verbose=1, period=1)

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

        print("SAVING WEIGHTS TO " + weights_file)

        model.save_weights(weights_file, True)
        print("TRAINING COMPLETE!")
        print("LOSS: {:5.2f}".format(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))

        return history

    else:
        print("LAST TRAINING DATA NOT FOUND")
        exit(3)


def train(dataset, num_tran_examples, num_val_examples):
    print("START NEW TRAINING")
    1
    if os.path.isdir(train_dir):
        os.system("rm -rf " + train_dir)

    captions_file_path, images_dir_path = Dataset.download_dataset(dataset)

    # load captions from dataset
    train_captions = Dataset.load_train_captions(dataset, num_tran_examples)
    train_captions = clean_captions(train_captions)
    train_images_name_list = Dataset.load_images_name(dataset, train_captions.keys())
    train_captions = add_start_end_token(train_captions)
    train_captions_list = to_captions_list(train_captions)

    val_captions = Dataset.load_eval_captions(dataset, num_val_examples)
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
    with open(dataset_name_file, "w") as f:
        f.write(dataset.get_name())

    store_vocabulary(vocabulary, word_index_dict, index_word_dict, vocabulary_dir, max_cap_len)
    store_train_data(train_dir, train_captions, train_images_as_vector)
    save_model(model, model_file)
    store_val_data(train_dir, val_captions, val_images_as_vector)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    # callbacks
    save_weights_callback = ModelCheckpoint(weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)
    save_epoch_callback = EpochSaver(1)
    save_model_callback = ModelCheckpoint(model_file, verbose=1, period=1)

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

    print("SAVING WEIGHTS TO " + weights_file)

    model.save_weights(weights_file, True)
    print("TRAINING COMPLETE!")
    print("LOSS: {:5.2f}".format(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))

    return history


def predict(image_name):
    vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(vocabulary_dir)

    print("VOCABULARY SIZE: " + str(len(vocabulary)))
    print("MAX CAPTION LENGTH: " + str(max_cap_len))

    model = create_NN(len(vocabulary), max_cap_len)

    model.load_weights(weights_file)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    predicted_caption = predict_caption(model, image_name, max_cap_len,
                                        word_index_dict, index_word_dict)

    print(predicted_caption)

    return predicted_caption


## START PROGRAM

if __name__ == "__main__":

    # check args
    if len(sys.argv) < 2:
        usage()
        exit(1)

    mode = sys.argv[1]

    # default values
    dataset_name = "flickr"
    num_tran_examples = 0
    num_val_examples = 0
    image_file_name = ""

    if mode == "train":
        num_args = 2 + (2 * 3)
        num_tran_examples = 6000
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
                    num_tran_examples = int(val)

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

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    dataset = Dataset.create_dataset(dataset_name)

    # select mode
    if mode == "train":

        hystory = train(dataset, num_tran_examples, num_val_examples)


    elif mode == "resume":
        hystory = resume()


    elif mode == "predict":

        predicted_caption = predict(image_file_name)

    exit(0)
