import numpy as np
import os
import json
import random

from model import COCODataset, Dataset
from process_data import preprocess_images, clean_captions, \
    generate_vocabulary, to_captions_list, add_start_end_token, load_all_images_name

data_path = "./data/"

if not os.path.isdir(data_path):
    os.makedirs(data_path)



dataset=COCODataset()




caption_file_path, dataset_dir_path = Dataset.download_dataset(dataset)


all_captions = Dataset.load_captions(dataset)  # dict image_id - caption

all_images_name_list = load_all_images_name(dataset_dir_path)

# Shuffle captions
l = list(all_captions.items())
random.shuffle(l)
all_captions = dict(l)

# Select the first 30000 captions from the shuffled set
num_training_examples = 3000
num_training_examples = 3

train_captions = dict(list(all_captions.items())[:num_training_examples])

train_images_name_list = Dataset.load_images_name(dataset, train_captions.keys())

random.shuffle(all_images_name_list)

train_captions_cleaned = clean_captions(train_captions)
train_captions_list = to_captions_list(train_captions_cleaned)

max_cap_len = max(len(d.split()) for d in train_captions_list)

vocabulary = generate_vocabulary(train_captions_list)

train_captions_list = add_start_end_token(train_captions_list)

train_images_as_vector = preprocess_images(dataset_dir_path, train_images_name_list, "train_images_vector.json")
