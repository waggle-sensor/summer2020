# WeatherNet V2

## 1. get_data_bash
Run this script to get data off of @bebop.lcrc.anl.gov. Two files are returned: flir and image. Create a new dir called lcrc_data and move both flir and image 
into lcrc_data.

<br>

## 2. weather_data_15min
The directory named weather_data_15min containts a folder called 15_min_data (this folder has the original weather data sampled at 15 minute intervals),
a script named parse_15_min_data.ipynb. Run the .ipynb file and a csv file named 15_min_weather_prep.csv is created and an image named Average_Net_Radition_Values_pic.jpg.
The file 15_min_weather_prep.csv will be used for the experiments. The image Average_Net_Values_pic.jpg shows the distribution of radiation values and the
three categories the values get put into.


![tree](WeatherNet_v2_Images/Average_Net_Radiation_Values_pic.jpg)    

<br>
