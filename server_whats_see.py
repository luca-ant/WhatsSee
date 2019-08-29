import sys
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

import whats_see

PORT = 4753

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

app = Flask(__name__)
socketio = SocketIO(app)


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
    resume = True
    running = True
    emit('state', {'resume': resume, 'running': running}, broadcast=True)


@socketio.on('resume', namespace='/message')
def resume():
    # create and start thread

    log = "resume"
    emit('log', {'data': log}, broadcast=True)


@socketio.on('start', namespace='/message')
def start(message):
    dataset_name = message['dataset']
    num_train_examples = int(message['nt'])
    num_val_examples = int(message['nv'])

    whatssee = whats_see.WhatsSee.get_instance()
    whatssee.set_dataset(dataset_name)

    # create and start thread

    log = "start"
    emit('log', {'data': log}, broadcast=True)


@socketio.on('stop', namespace='/message')
def stop():
    # stop thread
    log = "stop"

    emit('log', {'data': log}, broadcast=True)


@socketio.on('caption', namespace='/message')
def caption(message):
    whatssee = whats_see.WhatsSee.get_instance()
    filename = whatssee.captioned_images_dir + message['filename']
    caption = whatssee.predict(filename)
    emit('response', {'caption': caption}, broadcast=True)


if __name__ == "__main__":
    dataset_name = "flickr"
    whatssee = whats_see.WhatsSee(dataset_name, working_dir)

    whatssee.set_dataset(dataset_name)

    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
