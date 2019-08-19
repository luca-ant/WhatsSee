import os
import json
import random

from process_data import download_dataset, preprocess_images, load_captions, load_images_name

data_path = "./data/"

if not os.path.isdir(data_path):
    os.makedirs(data_path)

caption_file_path, dataset_dir_path = download_dataset()

print(caption_file_path, dataset_dir_path)







all_captions  = load_captions(caption_file_path)

all_images_name =load_images_name(dataset_dir_path)


# Shuffle captions and image_names together
train_captions, img_name_vector = random.shuffle(all_captions,
                                                 all_images_name,
                                                 random_state=1)

# Select the first 30000 captions from the shuffled set
num_training_examples = 30000
train_captions = train_captions[:num_training_examples]
train_images_name = img_name_vector[:num_training_examples]

train_images_as_vector = preprocess_images(dataset_dir_path, train_images_name, "train_images_vector.json")
