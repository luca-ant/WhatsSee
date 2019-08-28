import sys
import os

from flask import Flask, render_template

import whats_see
from model import Dataset

PORT = 4753

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

app = Flask(__name__)


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

    ws = whats_see.WhatsSee(dataset_name, working_dir)

    ws.dataset = Dataset.create_dataset(dataset_name, ws.data_dir)

    app.run(host='0.0.0.0', port=PORT, debug=True)
