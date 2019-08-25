import os
import sys

current_work_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

data_path = current_work_dir + "/data/"
vocabulary_path = current_work_dir + "/vocabulary/"
weight_path = current_work_dir + "/weights/"


def usage():
    print("Usage: " + sys.argv[0] + " [train | eval | predict] ")
    exit(1)


def usage_train():
    print("Usage: " + sys.argv[0] + " train dataset=[coco | flickr] num_example=NUMBER")
    exit(2)


def usage_eval():
    print("Usage: " + sys.argv[0] + " eval dataset=[coco | flickr]")
    exit(2)


def usage_predict():
    print("Usage: " + sys.argv[0] + " predict filename=YOUR_IMAGE_FILE")
    exit(2)


def train(dataset, num_training_examples):
    captions_file_path, images_dir_path = Dataset.download_dataset(dataset)

    train_captions = Dataset.load_train_captions(dataset, num_training_examples)
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

    store_vocabulary(vocabulary, word_index_dict, index_word_dict, vocabulary_path, max_cap_len)

    train_images_as_vector = preprocess_images(images_dir_path, train_images_name_list)

    model = create_NN(len(vocabulary), max_cap_len)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()

    # x_text, x_image, y_caption = prepare_data(dataset, train_captions, train_images_as_vector, word_index_dict, len(vocabulary), max_cap_len)

    # history = model.fit([x_image, x_text], y_caption, epochs=3, verbose=1,batch_size=16)

    epochs = 10
    num_photos_per_batch = 32
    steps = len(train_captions)
    for i in range(1, epochs + 1):
        print("EPOCH: " + str(i) + "/" + str(epochs))
        generator = data_generator(dataset, train_captions, train_images_as_vector, word_index_dict, max_cap_len,
                                   len(vocabulary), num_photos_per_batch)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    if not os.path.isdir(weight_path):
        os.makedirs(weight_path)

    model.save_weights(weight_path + "weights.h5", True)


def eval():
    vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(vocabulary_path)

    print("VOCABULARY SIZE: " + str(len(vocabulary)))
    print("MAX CAPTION LENGTH: " + str(max_cap_len))

    model = create_NN(len(vocabulary), max_cap_len)

    model.load_weights(weight_path + "weights.h5")


def predict(image_name):
    vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(vocabulary_path)

    print("VOCABULARY SIZE: " + str(len(vocabulary)))
    print("MAX CAPTION LENGTH: " + str(max_cap_len))

    model = create_NN(len(vocabulary), max_cap_len)

    model.load_weights(weight_path + "weights.h5")

    ### PREDICT

    predicted_caption = predict_caption(model, image_name, max_cap_len,
                                        word_index_dict, index_word_dict)

    print(predicted_caption)

    return predicted_caption


if len(sys.argv) < 2:
    usage()
    exit(1)

mode = sys.argv[1]

dataset_name = ""
num_training_examples = 1
image_file_name = ""

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

        elif key == "num_example":
            try:
                num_training_examples = int(val)

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

from model import COCODataset, FlickrDataset, Dataset, create_NN
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, \
    predict_caption, store_vocabulary, load_vocabulary, data_generator



if not os.path.isdir(data_path):
    os.makedirs(data_path)

if dataset_name == "coco":
    dataset = COCODataset(data_path)
elif dataset_name == "flickr":
    dataset = FlickrDataset(data_path)



if mode == "train":

    train(dataset, num_training_examples)





elif mode == "eval":
    eval()










elif mode == "predict":
    #    image_file_name = "COCO_train2014_000000000025.jpg"  # DEBUG
    #    image_file_name = "1067180831_a59dc64344.jpg"  # DEBUG
    predicted_caption = predict(image_file_name)

exit(0)
