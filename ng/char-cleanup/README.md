This script prepares the [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) dataset for training with YOLOv3.

To begin, create a folder (specified by `DATA` in the script) with the following subdirectories:

* `images`: contains extracted English letter images (not masks), with subfolders `GoodImag` and `BadIamg`. These subfolders each contain folders `Sample001` through `Sample062`, corresponding to 0-9, A-Z, and a-z (in that order)
* `labels`: empty folder that will contain labels

Afterwards, run `cleanup.py`. This will generate class names based on the most frequent classes within the dataset. The images belonging to these classes will then be split up into testing and training sets. All selected images will also be labeled.

The `augment.py` script can be used to augment the training dataset via the `albumentations` library. Run it with the command `python3 augment.py train.txt`. It has 8 filters, each run for 5 trials, which will augment a data set by 40x.

Upon finish, the script also creates labeled for the transformed image.