# WeatherNet V2

## 1. get_data_bash
Run this script to get data off of @bebop.lcrc.anl.gov. Two files are returned: flir and image. Create a new dir called lcrc_data and move both flir and image 
into lcrc_data.

<br>

## 2. Run parse_15_min_data.ipynb inside of weather_data_15min
The directory named weather_data_15min containts a folder called 15_min_data (this folder has the original weather data sampled at 15 minute intervals),
a script named parse_15_min_data.ipynb. Run the .ipynb file and a csv file named 15_min_weather_prep.csv is created and an image named Average_Net_Radition_Values_pic.jpg.
The file 15_min_weather_prep.csv will be used for the experiments. The image Average_Net_Values_pic.jpg shows the distribution of radiation values and the
three categories the values get put into.


![tree](WeatherNet_v2_Images/Average_Net_Radiation_Values_pic.jpg)    

<br>

## 3. Run prep_lcrc_data.py
This script will give create a new folder for the each type of image (flir, top, bottom) and move the respective files within. The files also get renamed
where the name is a timestamp. The script also removed images if the corresponding image from a particular time can't be found from all three cameras.
The new folders are stored in a new folder called data_clean.
<br>

## 4. Run tower data test within TowerDataTest 
Run Scale_Train_Test.ipynb to create two .csv files: train_weather.csv and test_weather.csv. Then run test_tower_data.ipynb to train and test model to foreacast 15 minute future solar values.
