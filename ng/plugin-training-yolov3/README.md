### Yolo V3 Training/Inferencing Plugin

The plugin runs Darknet-based Yolov3 model for both training and inferencing. In order to run the plugin user must have Docker engine (greater than 18.X.X) installed on the host. Nvidia CUDA driver (>= 10.1) on the host is preferrable for GPU acceleration.

1) Preparing Dataset

Image dataset is prepared on the host machine and the root path of the dataset will be mounted onto the plugin container. For training labeled images also need to exist. The following files need to be prepared as well.

- `class.names` is a file containing class names; one class name per line
- `train.txt` is a file listing path of the images for training; one image path per line; the path should start with `data/`
- `valid.txt` is a file listing path of the images for validation; one image path per line; the path should start with `data/`

2) Preparing Model Configuration

The plugin requires Yolov3 model configuration file as well as a data file that specifies path of the dataset files. Refer to [here](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master/config) for details.

3) Training

To train, simply run the command below on the host machine. Please make sure to set all the path correct.

```
# skip --runtime nvidia if the host is not CUDA accelerated
docker run -d --rm \
  --runtime nvidia \
  --name yolov3-training \
  -v ${PATH_TO_IMAGES}:/plugin/data \
  -v ${PATH_TO_CONFIG}:/plugin/config \
  -v ${PATH_TO_PRETRAINED_WEIGHTS}:/plugin/weights \
  -v ${PATH_FOR_OUTPUT_MODELS}:/plugin/checkpoints \
  -v ${PATH_TO_LOGS}:/plugin/logs \
  waggle/plugin-training-yolov3 \
  train \
  --epoch ${NUMBER_OF_EPOCH} \
  --batch_size ${BATCH_SIZE} \
  --model_def config/yolov3-custom.cfg \
  --data_config config/custom.data \
  --pretrained_weights weights/yolov3_ckpt_100.pth \
  --n_cpu ${NUMBER_OF_CPU} \
  --checkpoint_interval ${NUMBER_OF_CHK_INTERVAL} \
  --evaluation_interval ${NUMBER_OF_EVAL_INTERVAL}
```

The log of the training can be shown by,

```
docker logs -f yolov3-training
```

After the training is completed checkpoint models can be found in `${PATH_FOR_OUTPUT_MODELS}` on the host machine. The logs are saved in `${PATH_TO_LOGS}` and can be rendered by `tensorboard`.

```
$ tensorboard --logdir ${PATH_TO_LOGS}
```

4) Inferencing

__NOTE: This is not yet tested__

Prepare a set of images and trained model for inferencing. `class.names` used for training should be in the same folder with the images.

```
docker run -ti --rm \
  --runtime nvidia \
  --name yolov3-inferencing \
  -v ${PATH_TO_IMAGES}:/plugin/data/samples \
  -v ${PATH_TO_CONFIG}:/plugin/config \
  -v ${PATH_TO_TRAINED_MODEL}:/plugin/weights \
  -v ${PATH_FOR_INFERENCED_IMAGES}:/plugin/output \
  waggle/plugin-training-yolov3 \
  detect \
  --model_def config/yolov3-custom.cfg \
  --weights_path weights/${TRAINED_MODEL} \
  --class_path data/samples/class.names
```

After the inferencing is completed, `${PATH_FOR_INFERENCED_IMAGES}` should have images with the inferencing result.
