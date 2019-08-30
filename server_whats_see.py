import sys
import os
import threading
import traceback
from multiprocessing import Process
from time import sleep

import eventlet
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, send
from gevent import monkey

import whats_see

PORT = 4753

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

app = Flask(__name__)

monkey.patch_all()
sio = SocketIO(app, async_mode='gevent')

p = None


def start_training(num_train_examples, num_val_examples):
    global sio

    whatssee = whats_see.WhatsSee.get_instance()

    log = "CLEANING"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    t = threading.Thread(target=whatssee.clean_last_training_data)
    t.start()
    t.join()

    log = "DOWLOADING DATASET"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    t = threading.Thread(target=whatssee.download_dataset)
    t.start()
    t.join()

    log = "PROCESSING DATA"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    t = threading.Thread(target=whatssee.process_raw_data, args=(num_train_examples, num_val_examples,))
    t.start()
    t.join()

    log = "SAVE DATA"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    t = threading.Thread(target=whatssee.save_data_on_disk)
    t.start()
    t.join()

    log = "TRAINING IN PROGRESS"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    t = threading.Thread(target=whatssee.start_train)
    t.start()
    t.join()

    log = "END TRAINING"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)

    resume, running = get_state()
    running = False
    sio.emit('state', {'resume': resume, 'running': running}, namespace='/message', broadcast=True)

    # log = "CLEANING"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    # whatssee.clean_last_training_data()
    # log = "DOWLOADING DATASET"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    # whatssee.download_dataset()
    # log = "PROCESSING DATA"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    # whatssee.process_raw_data(num_train_examples, num_val_examples)
    # log = "SAVE DATA"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    # whatssee.save_data_on_disk()
    # log = "STARTING TRAINING"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    # whatssee.start_train()
    # log = "END TRAINING"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)


def resume_training():
    global sio
    whatssee = whats_see.WhatsSee.get_instance()

    log = "LOADING DATA"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    print(log)

    t = threading.Thread(target=whatssee.load_data_from_disk)
    t.start()

    t.join()

    log = "TRAINING IN PROGRESS"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    print(log)

    t = threading.Thread(target=whatssee.start_train)
    t.start()
    t.join()

    log = "END TRAINING"
    sio.emit('log', {'data': log}, namespace='/message', broadcast=True)

    resume, running = get_state()
    running = False
    sio.emit('state', {'resume': resume, 'running': running}, namespace='/message', broadcast=True)
    # log = "LOADING DATA"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    # whatssee.load_data_from_disk()
    # log = "RESUMING TRAINING"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)
    # whatssee.start_train()
    # log = "END TRAINING"
    # sio.emit('log', {'data': log}, namespace='/message', broadcast=True)


def get_state():
    whatssee = whats_see.WhatsSee.get_instance()

    res = False
    run = False

    if p != None and p.is_alive():
        run = True
    else:
        run = False

    if os.path.isdir(whatssee.train_dir):
        res = True

    else:
        res = False
    return res, run


@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/image", methods=['POST'])
def image():
    imagefile = request.files.get('imagefile', '')
    filename = request.form.get('filename')
    whatssee = whats_see.WhatsSee.get_instance()

    imagefile.save(whatssee.captioned_images_dir + filename)

    return "Image received!"


@sio.on('connect', namespace='/message')
def connect():
    resume, running = get_state()
    emit('state', {'resume': resume, 'running': running}, broadcast=True)

    if running:
        log = "TRAINING IN PROGRESS"
        emit('log', {'data': log}, namespace='/message', broadcast=True)


@sio.on('resume', namespace='/message')
def resume():
    global p
    p = Process(target=whatssee.resume)
    p.start()
    log = "\nRESUME TRAINING"
    emit('log', {'data': log}, namespace='/message', broadcast=True)
    resume, running = get_state()
    emit('state', {'resume': resume, 'running': running}, broadcast=True)


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
    log = "\nNEW TRAINING"
    emit('log', {'data': log}, namespace='/message', broadcast=True)
    resume, running = get_state()
    emit('state', {'resume': resume, 'running': running}, broadcast=True)


@sio.on('stop', namespace='/message')
def stop():
    global p
    p.kill()

    log = "STOP TRAINING"
    emit('log', {'data': log}, broadcast=True)

    resume, running = get_state()

    running = False
    emit('state', {'resume': resume, 'running': running}, broadcast=True)


@sio.on('caption', namespace='/message')
def caption(message):
    whatssee = whats_see.WhatsSee.get_instance()
    filename = whatssee.captioned_images_dir + message['filename']
    caption = whatssee.predict(filename)
    emit('response', {'caption': caption})


if __name__ == "__main__":
    dataset_name = "flickr"
    whatssee = whats_see.WhatsSee(dataset_name, working_dir)

    sio.run(app, host='0.0.0.0', port=PORT)
