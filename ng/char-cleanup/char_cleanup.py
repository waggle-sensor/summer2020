#!/usr/bin/env python3
import string
import os
import glob
import random

DATA = "/media/sng/My Passport/letters/"
OUTPUT = "./output/"

def generate_classes(out_file):
	classes = [str(i) for i in range(10)]
	for c in string.ascii_uppercase:
		classes.append(c)
	for c in string.ascii_lowercase:
		classes.append(c)

	with open(out_file, "w+") as out:
		out.write("\n".join(classes) + "\n")

def create_labels(img_paths):
	for path in img_paths:
		txt_path = path.replace("images", "labels").replace(".png", ".txt")

		folder_path = "/".join(txt_path.split("/")[:-1])
		os.makedirs(folder_path, exist_ok=True)

		idx = int(path.split("Sample0")[1][:2]) - 1
		line = f"{idx} 0.5 0.5 1 1"

		with open(txt_path, "w+") as out:
			out.write(line)

def split_test_train(img_paths, prop_train):
	random.shuffle(img_paths)

	train_num = int(prop_train * len(img_paths))
	train_list = img_paths[:train_num]
	test_list = img_paths[train_num:len(img_paths)]

	with open(OUTPUT + "train.txt", "w+") as out:
		out.write("\n".join(train_list))

	with open(OUTPUT + "test.txt", "w+") as out:
		out.write("\n".join(test_list))

def main():
	os.makedirs(OUTPUT, exist_ok=True)

	class_file = OUTPUT + "chars.names"
	if not os.path.exists(class_file):
		generate_classes(class_file)

	img_paths = glob.glob(DATA + "images/**/*.png", recursive=True)
	
	create_labels(img_paths)
	split_test_train(img_paths, 0.7)


if __name__ == '__main__':
	main()