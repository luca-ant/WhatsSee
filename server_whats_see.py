import ctypes
import sys
import os
import traceback
from multiprocessing import Process
from threading import Thread
from typing import Optional, Callable, Any, Iterable, Mapping

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

import whats_see

PORT = 4753

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

app = Flask(__name__)
socketio = SocketIO(app)

p = None


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


@socketio.on('connect', namespace='/message')
def connect():
    resume, running = get_state()
    emit('state', {'resume': resume, 'running': running}, broadcast=True)
    if running:
        log = "TRAINING ON PROGRESS"
        emit('log', {'data': log}, broadcast=True)


@socketio.on('resume', namespace='/message')
def resume():
    log = "RESUMING TRAINING"
    emit('log', {'data': log}, broadcast=True)
    global p
    p = Process(target=whatssee.resume)
    p.start()
    resume, running = get_state()
    emit('state', {'resume': resume, 'running': running}, broadcast=True)


@socketio.on('start', namespace='/message')
def start(message):
    dataset_name = message['dataset']
    num_train_examples = int(message['nt'])
    num_val_examples = int(message['nv'])
    whatssee = whats_see.WhatsSee.get_instance()
    whatssee.set_dataset(dataset_name)
    log = "STARTING TRAINING"
    emit('log', {'data': log}, broadcast=True)
    global p
    p = Process(target=whatssee.train, args=(num_train_examples, num_val_examples,))
    p.start()

    resume, running = get_state()
    emit('state', {'resume': resume, 'running': running}, broadcast=True)


@socketio.on('stop', namespace='/message')
def stop():
    global p
    p.kill()
    log = "STOPPING TRAINING"
    emit('log', {'data': log}, broadcast=True)

    resume, running = get_state()

    running = False
    emit('state', {'resume': resume, 'running': running}, broadcast=True)


@socketio.on('caption', namespace='/message')
def caption(message):
    whatssee = whats_see.WhatsSee.get_instance()
    filename = whatssee.captioned_images_dir + message['filename']
    caption = whatssee.predict(filename)
    emit('response', {'caption': caption})


if __name__ == "__main__":
    dataset_name = "flickr"
    whatssee = whats_see.WhatsSee(dataset_name, working_dir)

    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
