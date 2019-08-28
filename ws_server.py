import sys
import os
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from flask import Flask, render_template

import whats_see

PORT = 4753

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

app = Flask(__name__)

ws = whats_see.WhatsSee(working_dir)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/train")
def train():
    return render_template("train.html")


@app.route("/resume")
def resume():
    return "resume!"


@app.route("/predict")
def predict():
    return "predict!"


if __name__ == "__main__":
    app.run(port=PORT, debug=True)
