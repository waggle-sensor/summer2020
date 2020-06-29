#!/bin/bash

command=$1

usage() {
  echo "Usage:"
  echo "  To train: train OPTIONS"
  echo "     OPTIONS:"
  echo "           --epoch               : number of epoch"
  echo "           --batch_size          : batch size"
  echo "           --model_def           : model definition file"
  echo "           --data_config         : data configuration"
  echo "           --pretrained_weights  : pretrained model file"
  echo "           --n_cpu               : number of cpu"
  echo "           --checkpoint_interval : interval between saving model weights"
  echo "           --evaluation_interval : interval evaluations on validation set"

  echo "  To detect: detect OPTIONS"
  echo "     OPTIONS:"
  echo "           --image_folder               : path to dataset"
  echo "           --model_def                  : path to model definition file"
  echo "           --weights_path               : path to weights file"
  echo "           --class_path                 : path to class label file"
}

if [ "${command}x" == "trainx" ]; then
  python3 train.py ${@:2}
elif [ "${command}x" == "detectx" ]; then
  python3 detect.py ${@:2}
else
  usage
  exit 1
fi

# exit $?
