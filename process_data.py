import os
import json
import collections
import pickle
import string

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


def add_start_end_token(all_captions):
    all_captions_token = collections.defaultdict(list)

    for key, cap_list in all_captions.items():

        for c in cap_list:
            new_c = 'start_seq ' + c + ' end_seq'
            all_captions_token[key].append(new_c)

    return all_captions_token


def generate_vocabulary(all_captions_list):
    print("GENERATE VOCABULARY")

    vocabulary = []
    for c in all_captions_list:
        vocabulary = vocabulary + c.split()

    return list(set(vocabulary))


def store_vocabulary(vocabulary_dir, vocabulary, word_index_dict, index_word_dict, max_cap_len):
    print("STORE VOCABULARY")

    if not os.path.isdir(vocabulary_dir):
        os.makedirs(vocabulary_dir)

    with open(vocabulary_dir + "vocabulary.json", 'w') as f:
        #        f.write(json.dumps(vocabulary, default=lambda x: x.__dict__))
        f.write(json.dumps(vocabulary))

    with open(vocabulary_dir + "word_index_dict.json", 'w') as f:
        f.write(json.dumps(word_index_dict))

    with open(vocabulary_dir + "index_word_dict.json", 'w') as f:
        f.write(json.dumps(index_word_dict))

    with open(vocabulary_dir + "max_cap_len.json", 'w') as f:
        #        f.write(json.dumps(max_cap_len, default=lambda x: x.__dict__))
        f.write(json.dumps(max_cap_len))


def load_vocabulary(vocabulary_dir):
    print("LOAD VOCABULARY")

    if not os.path.isdir(vocabulary_dir):
        os.makedirs(vocabulary_dir)
        print("Vocabulary NOT FOUND")
        return

    with open(vocabulary_dir + "vocabulary.json") as f:
        vocabulary = json.load(f)

    with open(vocabulary_dir + "word_index_dict.json") as f:
        word_index_dict = json.load(f)

    with open(vocabulary_dir + "index_word_dict.json") as f:
        index_word_dict = json.load(f, object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})

    with open(vocabulary_dir + "max_cap_len.json") as f:
        max_cap_len = json.load(f)

    return vocabulary, word_index_dict, index_word_dict, max_cap_len


def store_train_data(train_dir, train_captions, train_images_as_vector):
    print("STORE TRAIN DATA")

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    with open(train_dir + "train_captions.json", 'w') as f:
        f.write(json.dumps(train_captions))

    f = open(train_dir + "train_images_as_vector.pkl", 'wb')
    pickle.dump(train_images_as_vector, f)
    f.close()


def load_train_data(train_dir):
    print("LOAD TRAIN DATA")

    with open(train_dir + "train_captions.json") as f:
        train_captions = json.load(f)

    f = open(train_dir + "train_images_as_vector.pkl", "rb")
    train_images_as_vector = pickle.load(f)
    f.close()

    return train_captions, train_images_as_vector


def store_val_data(train_dir, val_captions, val_images_as_vector):
    print("STORE VAL DATA")

    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    with open(train_dir + "val_captions.json", 'w') as f:
        f.write(json.dumps(val_captions))

    f = open(train_dir + "val_images_as_vector.pkl", 'wb')
    pickle.dump(val_images_as_vector, f)
    f.close()


def load_val_data(train_dir):
    print("LOAD VAL DATA")

    with open(train_dir + "val_captions.json") as f:
        val_captions = json.load(f)

    f = open(train_dir + "val_images_as_vector.pkl", "rb")
    val_images_as_vector = pickle.load(f)
    f.close()

    return val_captions, val_images_as_vector


def preprocess_images(images_dir_path, train_images_name_list):
    print("PROCESSING IMAGES")
    images_as_vector = collections.defaultdict()
    modelvgg = VGG16(include_top=True)

    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    # modelvgg.summary()

    with progressbar.ProgressBar(max_value=len(train_images_name_list)) as bar:
        for i, image_name in enumerate(train_images_name_list):
            img = image.load_img(images_dir_path + image_name, target_size=(224, 224, 3))
            img = image.img_to_array(img)

            img = preprocess_input(img)
            img_pred = modelvgg.predict(img.reshape((1,) + img.shape[:3]))
            images_as_vector[image_name] = img_pred.flatten()
            bar.update(i)

    return images_as_vector


def prepare_data(dataset, val_captions, val_images_as_vector, word_index_dict, vocab_size, max_cap_len):
    x_text, x_image, y_caption = [], [], []

    for image_id, cap_list in val_captions.items():

        image_name = Dataset.get_image_name(dataset, image_id)
        image = val_images_as_vector[image_name]

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
    modelvgg = VGG16(include_top=True)

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
