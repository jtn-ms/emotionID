#!/bin/bash
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
dir=$(zenity --file-selection --directory)
cd $DIR

echo $DIR

echo $dir

echo "Downloading..."

wget -O ./landmark-68.dat --continue --no-check-certificate "https://www.dropbox.com/s/4e420pi77fwddv6/landmark-68.dat?dl=0"
