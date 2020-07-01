#!/usr/bin/env python3
from __future__ import unicode_literals

import wget
import zipfile
import glob
import os


wget.download("http://www.iapr-tc11.org/dataset/KAIST_SceneText/KAIST_all.zip", ".")

zip_list = ["KAIST_all.zip"]

while len(zip_list) > 0:
    for zip_file in zip_list:
        with zipfile.ZipFile(zip_file, "r") as zip_obj:
            loc = zip_file[:-4]
            os.makedirs(loc, exist_ok=True)
            zip_obj.extractall(path=loc)
        os.remove(zip_file)
    zip_list = glob.glob("./**/*.zip", recursive=True)

os.rename("./KAIST_all/KAIST", "./images")
os.rmdir("./KAIST_all")

for img in glob.glob("./**/*.bmp", recursive=True):
    os.remove(img)
