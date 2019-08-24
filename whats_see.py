import os
import sys


def usage():
    print("Usage: " + sys.argv[0] + " [train | eval | predict] ")
    exit(1)


def usage_train():
    print("Usage: " + sys.argv[0] + " train dataset=[coco | flickr]")
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
    num_args = 2 + 1

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

from model import COCODataset, FlickrDataset, Dataset, create_NN
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, \
    predict_caption, store_vocabulary, load_vocabulary, data_generator

if dataset_name == "coco":
    dataset = COCODataset()
elif dataset_name == "flickr":
    dataset = FlickrDataset()

current_work_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

data_path = current_work_dir + "/data/"
vocabolary_path = current_work_dir + "/vocabulary/"
weight_path = current_work_dir + "/weight/"

if not os.path.isdir(data_path):
    os.makedirs(data_path)

if mode == "train":

    caption_file_path, dataset_dir_path = Dataset.download_dataset(dataset, data_path)

    train_captions = Dataset.load_train_captions(dataset)
    train_images_name_list = Dataset.load_images_name(dataset, train_captions.keys())

    train_captions_list = to_captions_list(train_captions)

    train_captions_list = add_start_end_token(train_captions_list)

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

    store_vocabulary(vocabulary, word_index_dict, index_word_dict, vocabolary_path)

    train_images_as_vector = preprocess_images(dataset_dir_path, train_images_name_list)

    model = create_NN(len(vocabulary), max_cap_len)

    x_text, x_image, y_caption = prepare_data(dataset, train_captions, train_images_as_vector, word_index_dict,
                                              len(vocabulary),
                                              max_cap_len)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()

    #  history = model.fit([x_image, x_text], y_caption, epochs=3, verbose=1,batch_size=16)

    epochs = 10
    num_photos_per_batch = 16
    steps = len(train_captions)
    for i in range(epochs):
        print("EPOCH: "+str(i)+"/"+str(epochs))
        generator = data_generator(dataset, train_captions, train_images_as_vector, word_index_dict, max_cap_len,
                                   len(vocabulary), num_photos_per_batch)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    if not os.path.isdir(weight_path):
        os.makedirs(weight_path)

    model.save_weights(weight_path + "weight.h5", True)

    ### PREDICT

    predicted_caption = predict_caption(model, dataset_dir_path, "COCO_train2014_000000000025.jpg", max_cap_len,
                                        word_index_dict, index_word_dict)

    print(predicted_caption)


elif mode == "eval":
    print(mode)
    caption_file_path, dataset_dir_path = Dataset.download_dataset(dataset, data_path)

    all_captions = Dataset.load_captions(dataset)  # dict image_id - caption










elif mode == "predict":
    print(mode)

exit(0)
