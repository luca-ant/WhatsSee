import datetime
import sys
import os
import threading
import traceback
import warnings
from multiprocessing import Process

import nltk as nltk
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, send
from gevent import monkey
from nltk.translate.bleu_score import sentence_bleu

import whats_see

warnings.simplefilter("ignore")

PORT = 4753

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

app = Flask(__name__)

monkey.patch_all()
sio = SocketIO(app, async_mode='gevent')

p = None
evaluation_in_progress = False


def logger(message):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y/%m/%d - %H:%M:%S")

    return timestamp + " " + message


def get_state():
    global evaluation_in_progress

    whatssee = whats_see.WhatsSee.get_instance()

    res = False
    run = False

    if p != None and p.is_alive():
        run = True

    if os.path.isdir(whatssee.train_dir):
        res = True

    return res, run, evaluation_in_progress


def start_training(num_train_examples, num_val_examples):
    global sio

    whatssee = whats_see.WhatsSee.get_instance()

    log = logger("CLEANING")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    t = threading.Thread(target=whatssee.clean_last_training_data)
    t.start()
    t.join()

    log = logger("DOWLOADING DATASET")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    t = threading.Thread(target=whatssee.download_dataset)
    t.start()
    t.join()

    log = logger("DONE!")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)

    log = logger("PROCESSING DATA")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    t = threading.Thread(target=whatssee.process_raw_data, args=(num_train_examples, num_val_examples,))
    t.start()
    t.join()

    log = logger("SAVING DATA")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    t = threading.Thread(target=whatssee.save_data_on_disk)
    t.start()
    t.join()

    log = logger("TRAINING IN PROGRESS")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)

    # t = threading.Thread(target=whatssee.start_train)
    # t.start()
    # t.join()

    history = whatssee.start_train()

    resume, running, evalrun = get_state()
    running = False
    sio.emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, namespace='/message', broadcast=True)

    log = logger("END TRAINING")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)

    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    acc = history.history['acc'][-1]
    val_acc = history.history['val_acc'][-1]

    log = "LOSS: {:5.2f}".format(loss) + "\nACC: {:5.2f}%".format(100 * acc) + "\nVAL_LOSS: {:5.2f}".format(val_loss) + "\nVAL_ACC: {:5.2f}%".format(100 * val_acc)
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)

    # log = "CLEANING"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # whatssee.clean_last_training_data()
    # log = "DOWLOADING DATASET"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # whatssee.download_dataset()
    # log = "PROCESSING DATA"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # whatssee.process_raw_data(num_train_examples, num_val_examples)
    # log = "SAVE DATA"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # whatssee.save_data_on_disk()
    # log = "TRAINING IN PROGRESS"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # whatssee.start_train()
    # log = "END TRAINING"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # resume, running = get_state()
    # running = False
    # sio.emit('state', {'resume': resume, 'running': running}, namespace='/message', broadcast=True)


def resume_training():
    global sio

    whatssee = whats_see.WhatsSee.get_instance()

    log = logger("LOADING DATA")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    t = threading.Thread(target=whatssee.load_data_from_disk)
    t.start()
    t.join()

    log = logger("TRAINING IN PROGRESS")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # t = threading.Thread(target=whatssee.start_train)
    # t.start()
    # t.join()
    history = whatssee.start_train()

    resume, running, evalrun = get_state()
    running = False
    sio.emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, namespace='/message', broadcast=True)

    log = logger("END TRAINING")
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)

    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    acc = history.history['acc'][-1]
    val_acc = history.history['val_acc'][-1]

    log = "LOSS: {:5.2f}".format(loss) + "\nACC: {:5.2f}%".format(100 * acc) + "\nVAL_LOSS: {:5.2f}".format(val_loss) + "\nVAL_ACC: {:5.2f}%".format(100 * val_acc)
    sio.emit('log', {'data': log}, namespace='/log', broadcast=True)

    # log = "LOADING DATA"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # whatssee.load_data_from_disk()
    # log = "RESUMING TRAINING"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)
    # whatssee.start_train()
    # log = "END TRAINING"
    # sio.emit('log', {'data': log}, namespace='/log', broadcast=True)


@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/image", methods=['POST'])
def image():
    imagefile = request.files.get('imagefile', '')
    filename = request.form.get('filename')
    ext = filename.rsplit(".", 1)[1]

    whatssee = whats_see.WhatsSee.get_instance()
    if ext.upper() in ["JPEG", "JPG", "PNG"]:
        imagefile.save(whatssee.captioned_images_dir + filename)
        return "Image received!"
    else:
        return "Invalid image!"


@sio.on('caption', namespace='/test')
def test_caption(message):
    whatssee = whats_see.WhatsSee.get_instance()
    filename = whatssee.dataset.test_images_dir + message['filename']
    caption = whatssee.predict(filename)
    original_captions = whatssee.dataset.get_captions_of(message['filename'])

    if original_captions:
        reference = []
        bleu_scores = []

        for c in original_captions:
            # reference.append(c.split())
            bleu_scores.append(sentence_bleu([c.strip().split()], caption.strip().split(), weights=(1.0, 0, 0, 0)))

        bleu_scores_string = []
        for b in bleu_scores:
            bleu_scores_string.append("BLEU SCORE: {:4.1f}%".format(100 * b))

        emit('response', {'caption': caption, 'originalcaptions': original_captions, 'bleuscores': bleu_scores_string})
    else:
        emit('response', {'caption': caption})


@sio.on('eval', namespace='/message')
def evaluate(message):
    num_examples = int(message['n'])
    global evaluation_in_progress
    evaluation_in_progress = True
    resume, running, evalrun = get_state()
    emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, broadcast=True)

    whatssee = whats_see.WhatsSee.get_instance()

    evaluation = whatssee.evaluate_model(num_examples)
    emit('evaluation', {'evaluation': evaluation}, broadcast=True)

    evaluation_in_progress = False
    resume, running, evalrun = get_state()
    emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, broadcast=True)



@sio.on('get', namespace='/test')
def get_test_images(message):
    whatssee = whats_see.WhatsSee.get_instance()
    whatssee.set_dataset(message["dataset"])
    test_images_name = whatssee.dataset.get_test_image_names()
    emit('images', {'images': test_images_name})


@sio.on('connect', namespace='/message')
def connect():
    resume, running, evalrun = get_state()
    emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, broadcast=True)

    if running:
        log = logger("TRAINING IN PROGRESS")
        emit('log', {'data': log}, namespace='/log', broadcast=True)


@sio.on('resume', namespace='/message')
def resume():
    global p
    p = Process(target=resume_training, args=())
    p.start()

    log = logger("RESUMING TRAINING")
    emit('log', {'data': log}, namespace='/log', broadcast=True)
    resume, running, evalrun = get_state()
    emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, broadcast=True)


@sio.on('start', namespace='/message')
def start(message):
    dataset_name = message['dataset']
    num_train_examples = int(message['nt'])
    num_val_examples = int(message['nv'])
    total_epochs = int(message['ne'])
    whatssee = whats_see.WhatsSee.get_instance()
    whatssee.set_dataset(dataset_name)
    whatssee.set_total_epochs(total_epochs)

    global p
    p = Process(target=start_training, args=(num_train_examples, num_val_examples,))
    p.start()

    log = logger("STARTING NEW TRAINING")
    emit('log', {'data': log}, namespace='/log', broadcast=True)
    resume, running, evalrun = get_state()
    emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, broadcast=True)


@sio.on('stop', namespace='/message')
def stop():
    global p
    p.kill()

    log = logger("STOP TRAINING")
    emit('log', {'data': log}, broadcast=True)

    resume, running, evalrun = get_state()
    running = False
    emit('state', {'resume': resume, 'running': running, 'evalrun': evalrun}, broadcast=True)


@sio.on('caption', namespace='/message')
def caption(message):
    whatssee = whats_see.WhatsSee.get_instance()
    filename = whatssee.captioned_images_dir + message['filename']
    caption = whatssee.predict(filename)
    emit('response', {'caption': caption})


if __name__ == "__main__":
    dataset_name = "flickr"
    whatssee = whats_see.WhatsSee(dataset_name, working_dir)

    sio.run(app, host='0.0.0.0', port=PORT, debug=True)
