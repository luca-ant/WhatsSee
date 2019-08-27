#!/usr/bin/python

import os
import sys
import logging
import traceback

from model import COCODataset, FlickrDataset, Dataset, create_NN
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, \
    predict_caption, store_vocabulary, load_vocabulary, data_generator, clean_captions
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

current_work_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

data_dir = current_work_dir + "/data/"
vocabulary_dir = data_dir + "vocabulary/"
weights_dir = data_dir + "weights/"

weights_file = weights_dir + "weights.h5"


def usage():
    print("Usage: " + sys.argv[0] + " [train | eval | predict] ")
    exit(1)


def usage_train():
    print("Usage: " + sys.argv[0] + " train -d [coco | flickr] -n NUMBER")
    exit(2)


def usage_eval():
    print("Usage: " + sys.argv[0] + " eval -d [coco | flickr] -n NUMBER")
    exit(2)


def usage_predict():
    print("Usage: " + sys.argv[0] + " predict -f YOUR_IMAGE_FILE")
    exit(2)


def train(dataset, num_training_examples):
    captions_file_path, images_dir_path = Dataset.download_dataset(dataset)

    train_captions = Dataset.load_train_captions(dataset, num_training_examples)
    train_captions = clean_captions(train_captions)
    train_images_name_list = Dataset.load_images_name(dataset, train_captions.keys())
    train_captions = add_start_end_token(train_captions)
    train_captions_list = to_captions_list(train_captions)

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

    store_vocabulary(vocabulary, word_index_dict, index_word_dict, vocabulary_dir, max_cap_len)

    train_images_as_vector = preprocess_images(images_dir_path, train_images_name_list)

    model = create_NN(len(vocabulary), max_cap_len)

    model.summary()

    if os.path.isfile(weights_file):
        try:
            model.load_weights(weights_file)
            print("LOAD PRE-TRAINED WEIGHTS OK")

        except:
            print("IMPOSSIBLE TO LOAD PRE-TRAINED WEIGHTS")
            # traceback.print_exc()
            pass

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # x_text, x_image, y_caption = prepare_data(dataset, train_captions, train_images_as_vector, word_index_dict, len(vocabulary), max_cap_len)

    # history = model.fit([x_image, x_text], y_caption, epochs=3, verbose=1,batch_size=16)

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    # save_weights_callback = ModelCheckpoint(weights_dir + "weights-epoch_{epoch:03d}.h5", monitor='val_acc',save_weights_only=True, verbose=1, mode='auto',period=1)

    save_weights_callback = ModelCheckpoint(weights_file, monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', period=1)

    num_images_per_batch = 32
    steps = len(train_captions)

    generator = data_generator(dataset, train_captions, train_images_as_vector, word_index_dict, max_cap_len,
                               len(vocabulary), num_images_per_batch)

    history = model.fit_generator(generator, epochs=5, steps_per_epoch=steps, verbose=1, callbacks=[save_weights_callback])

    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]

    print("LOSS: {:5.2f}".format(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    print("SAVING WEIGHTS TO " + weights_file)

    model.save_weights(weights_file, True)
    return history


def eval():
    captions_file_path, images_dir_path = Dataset.download_dataset(dataset)

    eval_captions = Dataset.load_eval_captions(dataset, num_training_examples)
    #    eval_captions = clean_captions(eval_captions)
    eval_images_name_list = Dataset.load_images_name(dataset, eval_captions.keys())

    eval_captions = add_start_end_token(eval_captions)

    vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(vocabulary_dir)

    print("VOCABULARY SIZE: " + str(len(vocabulary)))
    print("MAX CAPTION LENGTH: " + str(max_cap_len))

    eval_images_as_vector = preprocess_images(images_dir_path, eval_images_name_list)

    model = create_NN(len(vocabulary), max_cap_len)

    model.load_weights(weights_file)
    #    model.load_weights(checkpoints_dir + "last_model_checkpoint.h5")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_images_per_batch = 32
    steps = len(eval_captions)

    generator = data_generator(dataset, eval_captions, eval_images_as_vector, word_index_dict, max_cap_len,
                               len(vocabulary), num_images_per_batch)
    loss, acc = model.evaluate_generator(generator, steps=steps, max_queue_size=10, workers=1,
                                         use_multiprocessing=False, verbose=1)

    print("LOSS: {:5.2f}".format(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))

    return loss, acc


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

    # CHECK ARGS
    if len(sys.argv) < 2:
        usage()
        exit(1)

    mode = sys.argv[1]

    dataset_name = "flickr"
    num_training_examples = 0
    image_file_name = ""

    if mode == "train":
        num_args = 2 + (2 * 2)
        num_training_examples = 6000
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

            elif op == "-n":
                try:
                    num_training_examples = int(val)

                except:
                    print("Invalid value's option: " + val)
                    usage_train()

            else:
                print("Invalid option: " + op)
                usage_train()



    elif mode == "eval":
        num_args = 2 + (2 * 2)
        num_training_examples = 1000
        num_args = min(num_args, len(sys.argv))

        #        if len(sys.argv) < num_args:
        #            usage_eval()

        for i in range(2, num_args, 2):
            op = sys.argv[i]
            val = sys.argv[i + 1]

            if op == "-d":
                if val == "coco" or val == "flickr":
                    dataset_name = val
                else:
                    print("Invalid value's option: " + val)
                    usage_eval()
            elif op == "-n":
                try:
                    num_training_examples = int(val)

                except:
                    print("Invalid value's option: " + val)
                    usage_eval()

            else:
                print("Invalid option: " + op)
                usage_eval()


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

    if dataset_name == "coco":
        dataset = COCODataset(data_dir)
    elif dataset_name == "flickr":
        dataset = FlickrDataset(data_dir)

    if mode == "train":

        train(dataset, num_training_examples)

    elif mode == "eval":
        loss, acc = eval()

    elif mode == "predict":

        predicted_caption = predict(image_file_name)

    exit(0)
