import os
import glob

working_dir = "/Volumes/Samsung_T5/WeatherNet/Make_NP_DataSet"
val_path = working_dir + "/data_npy/val"

npy_val_flir_files = [os.path.basename(file) for i,file \
                      in enumerate(glob.glob(val_path+"/flir/*.npy"))]
npy_val_top_files = [os.path.basename(file) for i,file \
                      in enumerate(glob.glob(val_path+"/top/*.npy"))]
npy_val_bottom_files = [os.path.basename(file) for i,file \
                      in enumerate(glob.glob(val_path+"/bottom/*.npy"))]

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

inter = intersection(npy_val_top_files, npy_val_bottom_files)

for i,name in enumerate(npy_val_top_files):
    if name not in inter:
        os.remove(val_path+"/top/"+name)
