#!/bin/bash

RP=$(realpath $0)
CURRENT_DIR=$(dirname $RP)
source "$CURRENT_DIR/venv/bin/activate"

python "$CURRENT_DIR/server_whats_see.py"
