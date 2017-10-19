#!/bin/bash
#step1(install prerequisties)
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
#step2(optional)
##mkvirtualenv py2_dlib
#step3(python dlib install)
pip install numpy
pip install scipy
pip install scikit-image
pip install dlib
#step4(Test)
