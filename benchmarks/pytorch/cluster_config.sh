#!/bin/bash
# Must be run from MLLess' root directory

export PYTHONPATH=$(pwd)
echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc

# install and update pip3
apt update
apt install -y python3-pip
pip3 install --upgrade pip

# requirements
pip3 install -r requirements.txt

git clone https://github.com/pablogs98/MLLess
