#/bin/bash

mkdir data
cd data
wget http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz
tar xvzf EnglishImg.tgz
mv English/Img images
cd images
mv GoodImg/Bmp/* GoodImg/
mv BadImag/Bmp/* BadImag/
rm -r BadImag/Msk/
rm -r GoodImg/Msk/
cd ../..
pip3 install requirements.txt