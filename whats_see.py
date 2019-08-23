import os
import sys
import json
import random




def usage():
    print("Usage: " + sys.argv[0] + " [train | eval | predict] ")
    exit(1)


def usage_train():
    print("Usage: " + sys.argv[0] + " train dataset=[coco | flickr] num_examples=INT_NUMBER")
    exit(2)


def usage_eval():
    print("Usage: " + sys.argv[0] + " eval dataset=[coco | flickr]")
    exit(2)


def usage_predict():
    print("Usage: " + sys.argv[0] + " predict filename=your_image_file")
    exit(2)


if len(sys.argv) < 2:
    usage()
    exit(1)

mode = sys.argv[1]

if mode == "train":
    num_args = 2 + 2

    if len(sys.argv) < num_args:
        usage_train()

    for i in range(2, num_args):
        a = sys.argv[i]
        key, val = a.split("=")

        if key == "dataset":
            if val == "coco" or val == "flickr":
                dataset_name = val
            else:
                print("Invalid value's option: " + val)
                usage_train()

        elif key=="num_examples":
            try:
                num_training_examples = int(val)
                if num_training_examples > 8000:
                    num_training_examples = 8000
            except:
                print("Invalid value's option: " + val)
                usage_train()
        else:
            print("Invalid option: " + key)
            usage_train()


elif mode == "eval":
    num_args = 2 + 1

    if len(sys.argv) < num_args:
        usage_eval()

    for i in range(2, num_args):
        a = sys.argv[i]
        key, val = a.split("=")

        if key == "dataset":
            if val == "coco" or val == "flickr":
                dataset_name = val
            else:
                print("Invalid value's option: " + val)
                usage_eval()

        else:
            print("Invalid option: " + key)
            usage_eval()


elif mode == "predict":
    num_args = 2 + 1

    if len(sys.argv) < num_args:
        usage_predict()

    for i in range(2, num_args):
        a = sys.argv[i]
        key, val = a.split("=")
        if key == "filename":
            image_file_name = val
            if not os.path.isfile(image_file_name):
                print("404 File Not Found:" + image_file_name)
                usage_predict()
        else:
            print("Invalid option: " + key)

            usage_predict()




from model import COCODataset, Dataset, create_NN, FlickrDataset
from process_data import preprocess_images, clean_captions, \
    generate_vocabulary, to_captions_list, add_start_end_token, load_all_images_name, prepare_data, predict_caption



if dataset_name == "coco":
    dataset = COCODataset()
elif dataset_name == "flickr":
    dataset = FlickrDataset()

current_work_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

data_path = current_work_dir + "/data/"

if not os.path.isdir(data_path):
    os.makedirs(data_path)



if mode == "train":
    print(mode)
    caption_file_path, dataset_dir_path = Dataset.download_dataset(dataset, data_path)


    all_captions = Dataset.load_captions(dataset)  # dict image_id - caption

    all_images_name_list = load_all_images_name(dataset_dir_path)

    # Shuffle captions
    l = list(all_captions.items())
    random.shuffle(l)
    all_captions = dict(l)

    num_training_examples = 10 # DEBUG

    train_captions = dict(list(all_captions.items())[:num_training_examples])

    train_images_name_list = Dataset.load_images_name(dataset, train_captions.keys())


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

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()

    hist = model.fit([x_image, x_text], y_caption,
                     epochs=3, verbose=2,
                     batch_size=16)

    print(hist)



    ### PREDICT

    predicted_caption = predict_caption(model, dataset_dir_path, "1002674143_1b742ab4b8.jpg", max_cap_len,
                                        word_index_dict, index_word_dict)

    print(predicted_caption)


elif mode == "eval":
    print(mode)
    caption_file_path, dataset_dir_path = Dataset.download_dataset(dataset, data_path)

    all_captions = Dataset.load_captions(dataset)  # dict image_id - caption

    all_images_name_list = load_all_images_name(dataset_dir_path)









elif mode == "predict":
    print(mode)





exit(0)