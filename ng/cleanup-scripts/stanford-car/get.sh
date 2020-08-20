mkdir data
cd data
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
tar xvzf car_ims.tgz
mv car_ims images
cd ..
python3 mat_parse.py