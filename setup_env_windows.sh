#!/bin/bash
venv_dir="$(pwd -P)/.venv/"
if test -d $venv_dir;
then
    echo "activating virtual environment"
    source .venv/Scripts/activate
else
    echo "creating virtual environment"
    python -m venv .venv
    echo "activating virtual environment"
    source .venv/Scripts/activate
    echo "$(where python)"
    echo "activated venv"
    py -m pip install --upgrade pip
    py -m pip install -r requirements.txt
fi
