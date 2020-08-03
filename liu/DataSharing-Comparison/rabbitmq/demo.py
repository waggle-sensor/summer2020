#!/usr/bin/env python

receive_file_name = "receive_time.log"
emit_file_name = "emit_time.log"
my_open = open(receive_file_name, 'r')
my_open_emit = open(emit_file_name, 'r') 

# calculate the FPS
dict = {}

for eachline in my_open:
    # print(float(eachline.strip()))
    key = int(float(eachline.strip()))
    dict[key] = dict.get(key, 0) + 1

print(dict)
my_open.close()