This script prepares the [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) dataset for training with YOLOv3.

To begin, create a folder (specified by `DATA` in the script) with the following subdirectories:

* `images`: contains extracted English letter images (not masks), with subfolders `GoodImag` and `BadIamg`. These subfolders each contain folders `Sample001` through `Sample062`, corresponding to 0-9, A-Z, and a-z (in that order)
* `labels`: empty folder that will contain labels