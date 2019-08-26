import os
import sys
import logging
import tensorflow as tf
from model import COCODataset, FlickrDataset, Dataset, create_NN
from process_data import preprocess_images, generate_vocabulary, to_captions_list, add_start_end_token, prepare_data, \
    predict_caption, store_vocabulary, load_vocabulary, data_generator, clean_captions
from keras.callbacks import ModelCheckpoint

tf.get_logger().setLevel(logging.ERROR)

current_work_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

data_path = current_work_dir + "/data/"
vocabulary_path = current_work_dir + "/vocabulary/"
weight_path = current_work_dir + "/weights/"
checkpoints_path = current_work_dir + "/checkpoints/"


def usage():
    print("Usage: " + sys.argv[0] + " [train | eval | predict] ")
    exit(1)


def usage_train():
    print("Usage: " + sys.argv[0] + " train dataset=[coco | flickr] num_example=NUMBER")
    exit(2)


def usage_eval():
    print("Usage: " + sys.argv[0] + " eval dataset=[coco | flickr] num_example=NUMBER")
    exit(2)


def usage_predict():
    print("Usage: " + sys.argv[0] + " predict filename=YOUR_IMAGE_FILE")
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

    store_vocabulary(vocabulary, word_index_dict, index_word_dict, vocabulary_path, max_cap_len)

    train_images_as_vector = preprocess_images(images_dir_path, train_images_name_list)

    model = create_NN(len(vocabulary), max_cap_len)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()

    # x_text, x_image, y_caption = prepare_data(dataset, train_captions, train_images_as_vector, word_index_dict, len(vocabulary), max_cap_len)

    # history = model.fit([x_image, x_text], y_caption, epochs=3, verbose=1,batch_size=16)

    if os.path.isdir(checkpoints_path):
        os.system("rm -rf " + checkpoints_path)
    os.mkdir(checkpoints_path)

    # checkpoints_callback = ModelCheckpoint(checkpoints_path + "model-{epoch:03d}-{acc:03f}.h5", monitor='val_acc', save_weights_only=True, verbose=1, mode='auto', save_best_only=True, period=5)

    checkpoints_callback = ModelCheckpoint(checkpoints_path + "last_model_checkpoint.h5", monitor='val_acc',
                                           save_weights_only=True, verbose=1, mode='auto', period=1)

    num_images_per_batch = 32
    steps = len(train_captions)

    generator = data_generator(dataset, train_captions, train_images_as_vector, word_index_dict, max_cap_len,
                               len(vocabulary), num_images_per_batch)

    model.fit_generator(generator, epochs=100, steps_per_epoch=steps, verbose=1, callbacks=[checkpoints_callback])

    if not os.path.isdir(weight_path):
        os.makedirs(weight_path)

    model.save_weights(weight_path + "weights.h5", True)


def eval():
    captions_file_path, images_dir_path = Dataset.download_dataset(dataset)

    eval_captions = Dataset.load_eval_captions(dataset, num_training_examples)
    eval_captions = clean_captions(eval_captions)
    eval_images_name_list = Dataset.load_images_name(dataset, eval_captions.keys())

    eval_captions = add_start_end_token(eval_captions)

    vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(vocabulary_path)

    print("VOCABULARY SIZE: " + str(len(vocabulary)))
    print("MAX CAPTION LENGTH: " + str(max_cap_len))

    eval_images_as_vector = preprocess_images(images_dir_path, eval_images_name_list)

    model = create_NN(len(vocabulary), max_cap_len)

    model.load_weights(weight_path + "weights.h5")
    #    model.load_weights(checkpoints_path + "last_model_checkpoint.h5")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()

    num_images_per_batch = 32
    steps = len(eval_captions)

    generator = data_generator(dataset, eval_captions, eval_images_as_vector, word_index_dict, max_cap_len,
                               len(vocabulary), num_images_per_batch)
    loss, acc = model.evaluate_generator(generator, steps=steps, max_queue_size=10, workers=1,
                                         use_multiprocessing=False, verbose=1)

    print("LOSS: " + str(loss) + " - ACCURACY: {:5.2f}%".format(100 * acc))


def predict(image_name):
    vocabulary, word_index_dict, index_word_dict, max_cap_len = load_vocabulary(vocabulary_path)

    print("VOCABULARY SIZE: " + str(len(vocabulary)))
    print("MAX CAPTION LENGTH: " + str(max_cap_len))

    model = create_NN(len(vocabulary), max_cap_len)

    model.load_weights(weight_path + "weights.h5")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()

    ### PREDICT

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
        num_args = 2 + 2

        if len(sys.argv) < num_args:
            usage_eval()

        for i in range(2, num_args):
            a = sys.argv[i]
            if "=" not in a:
                continue
            key, val = a.split("=")

            if key == "dataset":
                if val == "coco" or val == "flickr":
                    dataset_name = val
                else:
                    print("Invalid value's option: " + val)
                    usage_eval()
            elif key == "num_example":
                try:
                    num_training_examples = int(val)

                except:
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

        predicted_caption = predict(image_file_name)

    exit(0)
