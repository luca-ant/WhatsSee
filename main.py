import collections
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


from keras_applications.vgg16 import preprocess_input
from tensorflow import keras

# Helper libraries
import numpy as np

from process_data import download_dataset, load_all_images_name

checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

caption_file_path, dataset_dir_path = download_dataset()
all_images_name_list = load_all_images_name(dataset_dir_path)

num_training_examples = 3

train_images_name = all_images_name_list[:num_training_examples]

images_as_vector = collections.defaultdict()

os.chdir(dataset_dir_path)

for image_path in train_images_name:
    # Convert all the images to size 299x299 as expected by the
    # inception v3 model
    img = Image.open(image_path)

    img.show()

    # Convert PIL image to numpy array of 3-dimensions

    i = (np.asarray(img))

    print(type(i))
    print(i)

    i = np.reshape(i, i.shape[1])

    # Add one more dimension
    i = np.expand_dims(i, axis=0)
    # preprocess images using preprocess_input() from inception module
    i = preprocess_input(i)
    # reshape from (1, 2048) to (2048, )
    images_as_vector[image_path] = i

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        period=5)

    """
    model.fit(train_images, train_labels, epochs=20, callbacks = [cp_callback])
    
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    
    print('Test accuracy:', test_acc)
    
    predictions = model.predict(test_images)
    
    img = test_images[0]
    
    print(img.shape)
    
    
    
    
    
    
    
    
    # load latest weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    model.load_weights(latest)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    """
