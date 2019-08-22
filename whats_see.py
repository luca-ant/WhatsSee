import numpy as np
import os
import json
import random

from keras.losses import categorical_crossentropy
from keras.optimizers import adam

from model import COCODataset, Dataset, create_NN
from process_data import preprocess_images, clean_captions, \
    generate_vocabulary, to_captions_list, add_start_end_token, load_all_images_name, prepare_data, predict_caption

data_path = "./data/"

if not os.path.isdir(data_path):
    os.makedirs(data_path)

dataset = COCODataset()

caption_file_path, dataset_dir_path = Dataset.download_dataset(dataset)

all_captions = Dataset.load_captions(dataset)  # dict image_id - caption

all_images_name_list = load_all_images_name(dataset_dir_path)

# Shuffle captions
l = list(all_captions.items())
random.shuffle(l)
all_captions = dict(l)

# Select the first 30000 captions from the shuffled set
num_training_examples = 3000
num_training_examples = 300

train_captions = dict(list(all_captions.items())[:num_training_examples])

train_images_name_list = Dataset.load_images_name(dataset, train_captions.keys())

random.shuffle(all_images_name_list)

train_captions = clean_captions(train_captions)

train_captions_list = to_captions_list(train_captions)

max_cap_len = max(len(d.split()) for d in train_captions_list)

train_captions_list = add_start_end_token(train_captions_list)

vocabulary = generate_vocabulary(train_captions_list)
print("VOCABULARY SIZE: " + str(len(vocabulary)))
vocabulary.append("0")

index_word_dict = {}
word_index_dict = {}
i = 1
for w in vocabulary:
    word_index_dict[w] = i
    index_word_dict[i] = w
    i += 1

train_images_as_vector = preprocess_images(dataset_dir_path, train_images_name_list)

model = create_NN(len(vocabulary), max_cap_len)

x_text, x_image, y_caption = prepare_data(dataset, train_captions, train_images_as_vector, word_index_dict,
                                          len(vocabulary),
                                          max_cap_len)

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

hist = model.fit([x_image, x_text], y_caption,
                 epochs=7, verbose=2,
                 batch_size=64)

### PREDICT

predicted_caption = predict_caption(model, dataset_dir_path, "COCO_train2014_000000000025.jpg", max_cap_len,
                                    word_index_dict, index_word_dict)

print(predicted_caption)
