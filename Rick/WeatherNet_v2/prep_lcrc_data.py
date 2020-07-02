
working_dir = "/Volumes/Samsung_T5/WeatherNet_V2"
data_dir = working_dir + "/lcrc_data"

#---------------------------------#

import glob
import shutil
import os

#---------------------------------#
os.mkdir(working_dir + "/data_clean")
os.mkdir(working_dir + "/data_clean/flir")
os.mkdir(working_dir + "/data_clean/top")
os.mkdir(working_dir + "/data_clean/bottom")

#---------------------------------#
flir_path = data_dir + "/flir/bottom/2020"
for root, dirs, files in os.walk(flir_path):
   for file in files:
      path_file = os.path.join(root,file)
      shutil.copy2(path_file,working_dir + "/data_clean/flir")

#---------------------------------#
flir_path = data_dir + "/image/bottom/2020"
for root, dirs, files in os.walk(flir_path):
   for file in files:
      path_file = os.path.join(root,file)
      shutil.copy2(path_file,working_dir + "/data_clean/bottom")

#---------------------------------#
flir_path = data_dir + "/image/top/2020"
for root, dirs, files in os.walk(flir_path):
   for file in files:
      path_file = os.path.join(root,file)
      shutil.copy2(path_file,working_dir + "/data_clean/top")

#---------------------------------#
def rename_pic(pic_name):
    pic_name  = pic_name.replace("T", "_")
    pic_name  = pic_name.replace("-", "_")
    pic_name  = pic_name.replace("/", "_")
    pic_name  = pic_name.replace(":", "_")
    pic_name = pic_name.split("+")[0]
    pic_name = pic_name + ".jpg"
    return pic_name

#---------------------------------#
for i, file in enumerate(glob.glob(working_dir + "/data_clean/flir/*")):
    pic_name = file.split("/")[-1]
    new_name = rename_pic(pic_name)
    new_name = new_name.split("_")[:5]
    new_name = "_".join(new_name)
    new_name = new_name + ".jpg"
    os.rename(file,working_dir + "/data_clean/flir/" + new_name)

#---------------------------------#
for i, file in enumerate(glob.glob(working_dir + "/data_clean/top/*")):
    pic_name = file.split("/")[-1]
    new_name = rename_pic(pic_name)
    new_name = new_name.split("_")[:5]
    new_name = "_".join(new_name)
    new_name = new_name + ".jpg"
    os.rename(file,working_dir + "/data_clean/top/" + new_name)

#---------------------------------#
for i, file in enumerate(glob.glob(working_dir + "/data_clean/bottom/*")):
    pic_name = file.split("/")[-1]
    new_name = rename_pic(pic_name)
    new_name = new_name.split("_")[:5]
    new_name = "_".join(new_name)
    new_name = new_name + ".jpg"
    os.rename(file,working_dir + "/data_clean/bottom/" + new_name)

#---------------------------------#
os.system("dot_clean .")
os.chdir(working_dir + "/data_clean/flir")
os.system("dot_clean .")
os.chdir(working_dir + "/data_clean/top")
os.system("dot_clean .")
os.chdir(working_dir + "/data_clean/bottom")
os.system("dot_clean .")
os.chdir(working_dir)

#---------------------------------#
flir_pics = glob.glob(working_dir + "/data_clean/flir/*.jpg")
bottom_pics = glob.glob(working_dir + "/data_clean/bottom/*.jpg")
top_pics = glob.glob(working_dir + "/data_clean/top/*.jpg")

flir_pics = [os.path.basename(file) for i,file in enumerate(flir_pics)]
top_pics = [os.path.basename(file) for i,file in enumerate(top_pics)]
bottom_pics = [os.path.basename(file) for i,file in enumerate(bottom_pics)]

flir_pics.sort()
bottom_pics.sort()
top_pics.sort()

#---------------------------------#
common_files = list(set(flir_pics).intersection(bottom_pics))
common_files = list(set(common_files).intersection(top_pics))
print("Number of common files: {}".format(len(common_files)))

for i ,file in enumerate(flir_pics):
    if file not in common_files:
        os.remove(working_dir + "/data_clean/flir/" + file)

for i ,file in enumerate(top_pics):
    if file not in common_files:
        os.remove(working_dir + "/data_clean/top/" + file)

for i ,file in enumerate(bottom_pics):
    if file not in common_files:
        os.remove(working_dir + "/data_clean/bottom/" + file)
