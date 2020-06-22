Within a central folder specified by `DATA`, there are three folders used for image cleaning in preparation for training with the `yolov3` plugin:

* `maingate_masks`: set of 3,340 alpha JPEG masks of vehicles. Has a `-car` or `-vehicle` suffix appended to the original filename
* `maingate_meta`: set of 7,166 XML files containing the metadata of images, with the following properties
  * `filename`
  * `folder`: path relative to the path specified by `DATA`
  * `object/name`: includes `useless`, `vehicle`, `crosswalk`, and `car`; those marked with `useless` have no attributes
  * `object/attributes`: a set of space-separated labels, with the format `<key>_<value> <key> confidence (0-100)_<conf. val.>` 
  	* Only certain keys have confidence values: `type`, `make`, `model`, `use`
* `waggle_maingate`: images and videos sourced from `/lcrc/project/waggle/public_html/private/training_data/waggle_maingate` on the LCRC server. Has 5,664 images in the `maingate_pic` subfolder and a combination of images and videos organized by date in the `waggle` subfolder