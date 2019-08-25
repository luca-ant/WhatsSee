import os
import json
import collections
import string
import random

import progressbar
import numpy as np

from model import Dataset

from keras import models
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from tensorflow.python.keras.preprocessing import image


def clean_captions(all_captions):
    all_captions_cleaned = {}
    punt = str.maketrans('', '', string.punctuation)

    for key, capt_list in all_captions.items():
        capt_list_cleaned = []
        for i in range(len(capt_list)):
            cap = capt_list[i]

            cap = cap.split()

            # convert to lower case
            cap = [word.lower() for word in cap]

            # remove punctuation from each token
            cap = [w.translate(punt) for w in cap]

            cap = [word for word in cap if len(word) > 1]

            # desc = [word for word in desc if word.isalpha()]
            capt_list_cleaned.append(' '.join(cap))

        all_captions_cleaned[key] = capt_list_cleaned

    return all_captions_cleaned


def to_captions_list(all_captions):
    all_train_captions = []
    for key, val in all_captions.items():
        for c in val:
            all_train_captions.append(c)
    return all_train_captions


def add_start_end_token(all_captions_list):
    all_captions_list_token = []
    for c in all_captions_list:
        new_c = 'start_seq ' + c + ' end_seq'
        all_captions_list_token.append(new_c)

    return all_captions_list_token


def generate_vocabulary(all_captions_list):
    print("GENERATE VOCABULARY")

    vocabulary = []
    for c in all_captions_list:
        vocabulary = vocabulary + c.split()

    return list(set(vocabulary))


def store_vocabulary(vocabulary, word_index_dict, index_word_dict, vocabulary_path, max_cap_len):
    print("STORE VOCABULARY")

    if not os.path.isdir(vocabulary_path):
        os.makedirs(vocabulary_path)

    with open(vocabulary_path + "vocabulary.json", 'w') as f:
        #        f.write(json.dumps(vocabulary, default=lambda x: x.__dict__))
        f.write(json.dumps(vocabulary))

    with open(vocabulary_path + "word_index_dict.json", 'w') as f:
        f.write(json.dumps(word_index_dict))

    with open(vocabulary_path + "index_word_dict.json", 'w') as f:
        f.write(json.dumps(index_word_dict))

    with open(vocabulary_path + "max_cap_len.json", 'w') as f:
        #        f.write(json.dumps(max_cap_len, default=lambda x: x.__dict__))
        f.write(json.dumps(max_cap_len))


def load_vocabulary(vocabulary_path):
    print("LOAD VOCABULARY")

    if not os.path.isdir(vocabulary_path):
        os.makedirs(vocabulary_path)
        print("Vocabulary NOT FOUND")

    with open(vocabulary_path + "vocabulary.json") as f:
        vocabulary = json.load(f)

    with open(vocabulary_path + "word_index_dict.json") as f:
        word_index_dict = json.load(f)

    with open(vocabulary_path + "index_word_dict.json") as f:
        index_word_dict = json.load(f)

    with open(vocabulary_path + "max_cap_len.json") as f:
        max_cap_len = json.load(f)

    return vocabulary, word_index_dict, index_word_dict, max_cap_len


def preprocess_images(dataset_dir_path, train_images_name):
    os.chdir(dataset_dir_path)

    images_as_vector = collections.defaultdict()

    #   modelvgg = VGG16(include_top=True, weights=None)
    modelvgg = VGG16(include_top=True)
    #       modelvgg.load_weights("../input/vgg16-weights-image-captioning/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    modelvgg.summary()

    with progressbar.ProgressBar(max_value=len(train_images_name)) as bar:
        for i, image_name in enumerate(train_images_name):
            img = image.load_img(image_name, target_size=(224, 224, 3))

            img = image.img_to_array(img)

            img = preprocess_input(img)
            img_pred = modelvgg.predict(img.reshape((1,) + img.shape[:3]))
            images_as_vector[image_name] = img_pred.flatten()
            bar.update(i)

    return images_as_vector


def prepare_data(dataset, train_captions, train_images_as_vector, word_index_dict, vocab_size, max_cap_len):
    x_text, x_image, y_caption = [], [], []

    for image_id, cap_list in train_captions.items():

        image_name = Dataset.get_image_name(dataset, image_id)
        image = train_images_as_vector[image_name]

        for c in cap_list:
            int_seq = [word_index_dict[word] for word in c.split(' ') if word in word_index_dict]

            for i in range(1, len(int_seq)):
                in_text, out_text = int_seq[:i], int_seq[i]

                in_text = pad_sequences([in_text], maxlen=max_cap_len)[0]
                out_text = to_categorical(out_text, num_classes=vocab_size)

                x_text.append(in_text)
                y_caption.append(out_text)
                x_image.append(image)

    x_text = np.array(x_text)
    x_image = np.array(x_image)
    y_caption = np.array(y_caption)
    print(" {} {} {}".format(x_text.shape, x_image.shape, y_caption.shape))
    return x_text, x_image, y_caption


def data_generator(dataset, train_captions, train_images_as_vector, word_index_dict, max_cap_len, vocab_size,
                   num_photos_per_batch):
    x_text, x_image, y_caption = list(), list(), list()
    n = 0
    while 1:

        for image_id, cap_list in train_captions.items():
            n += 1

            image_name = Dataset.get_image_name(dataset, image_id)
            image = train_images_as_vector[image_name]

            for c in cap_list:
                int_seq = [word_index_dict[word] for word in c.split(' ') if word in word_index_dict]

                for i in range(1, len(int_seq)):
                    in_text, out_text = int_seq[:i], int_seq[i]

                    in_text = pad_sequences([in_text], maxlen=max_cap_len)[0]
                    out_text = to_categorical(out_text, num_classes=vocab_size)

                    x_text.append(in_text)
                    y_caption.append(out_text)
                    x_image.append(image)

            if n == num_photos_per_batch:
                yield [[np.array(x_image), np.array(x_text)], np.array(y_caption)]
                x_text, x_image, y_caption = list(), list(), list()
                n = 0


def predict_caption(model, image_name, max_cap_len, word_index_dict, index_word_dict):
    # modelvgg = VGG16(include_top=True, weights=None)
    modelvgg = VGG16(include_top=True)
    #       modelvgg.load_weights("../input/vgg16-weights-image-captioning/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)

    img = image.load_img(image_name, target_size=(224, 224, 3))

    img = image.img_to_array(img)

    img = preprocess_input(img)
    img = modelvgg.predict(img.reshape((1,) + img.shape[:3]))

    in_text = 'start_seq'
    for i in range(max_cap_len):
        sequence = [word_index_dict[w] for w in in_text.split() if w in word_index_dict]
        sequence = pad_sequences([sequence], maxlen=max_cap_len)
        yhat = model.predict([img, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word_dict[yhat]
        in_text += ' ' + word
        if word == 'end_seq':
            break
    caption = in_text.split()
    caption = caption[1:-1]
    caption = ' '.join(caption)

    return caption
