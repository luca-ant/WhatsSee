import sys
import os

from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send

import whats_see
from model import Dataset

PORT = 4753

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

print(working_dir)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app)


@socketio.on('my event', namespace='/message')
def message(message):
    print(message)
    emit('my response', {'data': message['data']})


@socketio.on('broadcast', namespace='/message')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True)


@socketio.on('connect', namespace='/message')
def connect():
    send("connect")


@socketio.on('disconnect', namespace='/message')
def disconnect():
    send('Client disconnected')


@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/train", methods=['GET'])
def train():
    return "train!"


@app.route("/resume", methods=['GET'])
def resume():
    return "resume!"


@app.route("/predict", methods=['GET'])
def predict():
    return "predict!"


if __name__ == "__main__":
    dataset_name = "flickr"
    whatssee = whats_see.WhatsSee(dataset_name, working_dir)
    
    whatssee.set_dataset(dataset_name)

    socketio.run(app, host='0.0.0.0', port=PORT, debug=True)
