## Timeline analysis in TensorFlow

yolov3:
```
node name | requested bytes | total execution time | accelerator execution time | cpu execution time
Conv2D                      1015.29MB (100.00%, 61.17%),     215.96ms (100.00%, 74.32%),     170.59ms (100.00%, 82.80%),      45.37ms (100.00%, 53.65%)
LeakyRelu                             0B (0.00%, 0.00%),        11.97ms (25.68%, 4.12%),         5.42ms (17.20%, 2.63%),         6.56ms (46.35%, 7.75%)
FusedBatchNormV3                      0B (0.00%, 0.00%),         9.79ms (21.56%, 3.37%),         5.94ms (14.57%, 2.88%),         3.85ms (38.59%, 4.56%)
AddV2                                 0B (0.00%, 0.00%),         5.02ms (18.19%, 1.73%),         3.34ms (11.69%, 1.62%),         1.68ms (34.04%, 1.99%)
Tile                           114.69KB (38.83%, 0.01%),         4.35ms (16.46%, 1.50%),          838us (10.07%, 0.41%),         3.52ms (32.05%, 4.16%)
import/conv68/Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer        2.77MB (38.82%, 0.17%),         3.04ms (14.96%, 1.04%),            57us (9.66%, 0.03%),         2.98ms (27.89%, 3.52%)
ConcatV2                        10.51MB (38.65%, 0.63%),         2.54ms (13.92%, 0.88%),           947us (9.63%, 0.46%),         1.60ms (24.37%, 1.89%)
StridedSlice                     3.62MB (38.02%, 0.22%),         1.63ms (13.05%, 0.56%),           166us (9.17%, 0.08%),         1.46ms (22.48%, 1.73%)
```

