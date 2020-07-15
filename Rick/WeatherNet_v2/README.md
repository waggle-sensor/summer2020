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

![tree](WeatherNet_v2_Images/tower_test_data_results.png) 

### Forecast solar radiation with LSTM using tower data results

|                              |           |        |          |         | 
|------------------------------|-----------|--------|----------|---------|
| 15 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |  
| Low                          | 0.90      | 0.83   | 0.86     | 249     |  
| Mid                          | 0.49      | 0.67   | 0.56     | 106     |  
| High                         | 0.95      | 0.90   | 0.92     | 391     |     
| Accuracy                     |           |        | 0.84     | 746     |  
|                              |           |        |          |         |   
|                              |           |        |          |         |    
|                              |           |        |          |         |     
| 30 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |     
| Low                          | 0.96      | 0.87   | 0.91     | 249     |     
| Mid                          | 0.54      | 0.88   | 0.67     | 106     |     
| High                         | 0.98      | 0.87   | 0.93     | 391     |     
| Accuracy                     |           |        | 0.87     | 746     |     
|                              |           |        |          |         |     
|                              |           |        |          |         |     
| 45 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |     
| Low                          | 0.94      | 0.85   | 0.89     | 249     |     
| Mid                          | 0.51      | 0.73   | 0.60     | 106     |     
| High                         | 0.94      | 0.89   | 0.91     | 391     |     
| Accuracy                     |           |        | 0.85     | 745     |     
|                              |           |        |          |         |     
|                              |           |        |          |         |     
|                              |           |        |          |         |     
| 60 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |    
| Low                          | 0.86      | 0.80   | 0.83     | 249     |     
| Mid                          | 0.46      | 0.59   | 0.52     | 106     |     
| High                         | 0.94      | 0.90   | 0.92     | 391     |     
| Accuracy                     |           |        | 0.82     | 746     |     
|                              |           |        |          |         |     
|                              |           |        |          |         |     
| 75 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |    
| Low                          | 0.83      | 0.74   | 0.78     | 249     |     
| Mid                          | 0.38      | 0.57   | 0.45     | 106     |     
| High                         | 0.94      | 0.88   | 0.91     | 391     |     
| Accuracy                     |           |        | 0.79     | 746     |     
|                              |           |        |          |         |     
| 90 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |     
| Low                          | 0.78      | 0.70   | 0.74     | 249     |     
| Mid                          | 0.35      | 0.52   | 0.42     | 106     |     
| High                         | 0.95      | 0.88   | 0.91     | 391     |     
| Accuracy                     |           |        | 0.77     | 746     |     


<br>

## 5. Create img seq .npy files by running make_npy_samples.ipynb
In make_npy_samples.ipynb, select the length of the images sequences. Like the LSTM experiment that used four samples to form an hour long sequence, four ordered photos were used to make a sample. The script creates a dir called npy_dataset and make three dirs inside called top, bottom, and flir. The program places the npy sequence samples in the respective folder based on what type on the type of image.

<br>

## 6. Create scaled input and label dataframes for npy dataset for 15, 30, 45, 60, 75, and 90 min weather data.
This script will scale the data based on training size (training data goes up to the date, 2020_05_24_04_30). This is done by running make_weather_data_npy.ipynb.

<br> 

## 7. ImgSeqGen.ipynb
This notebook holds and demos the generator class that will be used my the model. It takes in data from a labels csv, a weather csv, and the three cameras and outputs shaped batches for the model.

<br>


## 7. Camera Data Test 
Run the WeatherNetV2_Model_Test.ipynb using the only the images, time, and previous period target values. The results of this experiment correspond to experiment two within the paper.

![tree](WeatherNet_v2_Images/WeatherNet_model.png) 


### Forecast solar radiation with ConvLSTM using images, time, and previous target values results

|                              |           |        |          |         | 
|------------------------------|-----------|--------|----------|---------|
| 15 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |  
| Low                          | 0.90      | 0.88   | 0.89     | 240     |  
| Mid                          | 0.78      | 0.79   | 0.78     | 165     |  
| High                         | 0.97      | 0.97   | 0.97     | 331     |     
| Accuracy                     |           |        | 0.90     | 736     |  
|                              |           |        |          |         |   
|                              |           |        |          |         |    
|                              |           |        |          |         |     
| 30 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |     
| Low                          | 0.82      | 0.87   | 0.84     | 209     |     
| Mid                          | 0.77      | 0.67   | 0.72     | 165     |     
| High                         | 0.95      | 0.98   | 0.96     | 266     |     
| Accuracy                     |           |        | 0.86     | 640     |   
|                              |           |        |          |         |     
|                              |           |        |          |         |     
| 45 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |     
| Low                          | 0.80      | 0.83   | 0.82     | 217     |     
| Mid                          | 0.72      | 0.64   | 0.68     | 168     |     
| High                         | 0.95      | 0.98   | 0.96     | 255     |     
| Accuracy                     |           |        | 0.84     | 640     |     
|                              |           |        |          |         |     
|                              |           |        |          |         | 
|                              |           |        |          |         |     
| 60 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |    
| Low                          | 0.78      | 0.79   | 0.78     | 222     |     
| Mid                          | 0.67      | 0.63   | 0.65     | 174     |     
| High                         | 0.94      | 0.97   | 0.96     | 244     |     
| Accuracy                     |           |        | 0.81     | 640     |     
|                              |           |        |          |         |     
|                              |           |        |          |         |   
| 75 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |    
| Low                          | 0.76      | 0.72   | 0.74     | 228     |     
| Mid                          | 0.61      | 0.63   | 0.62     | 178     |     
| High                         | 0.94      | 0.97   | 0.95     | 234     |     
| Accuracy                     |           |        | 0.79     | 640     |     
|                              |           |        |          |         |    
| 90 Minutes: Input length = 4 | Precision | Recall | F1-Score | Support |     
| Low                          | 0.75      | 0.67   | 0.71     | 232     |     
| Mid                          | 0.56      | 0.65   | 0.60     | 184     |     
| High                         | 0.94      | 0.92   | 0.93     | 224     |     
| Accuracy                     |           |        | 0.75     | 640     | 


<br>


## 8. Camera and Weather Tower Data Test


### Forecast solar radiation with ConvLSTM using images and tower data

|15m           |precision    |recall  |f1-score   |support|
|--------------|-------------|--------|-----------|-------|
|              |             |        |           |       |
|         low  |0.85         |0.88    |0.87       |201    |
|         mid  |0.75         |0.75    |0.75       |162    |
|        high  |0.96         |0.94    |0.95       |277    |
|              |             |        |           |       |
|    accuracy  |             |        |0.87       |640    |
|   macro avg  |0.85         |0.86    |0.85       |640    |
|weighted avg  |0.87         |0.87    |0.87       |640    |
|              |             |        |           |       |
|              |             |        |           |       |
|              |             |        |           |       |
|30m           |precision    |recall  |f1-score   |support|
|              |             |        |           |       |
|         low  |0.82         |0.83    |0.82       |209    |
|         mid  |0.67         |0.70    |0.68       |165    |
|        high  |0.96         |0.92    |0.94       |266    |
|              |             |        |           |       |
|    accuracy  |             |        |0.83       |640    |
|   macro avg  |0.81         |0.82    |0.81       |640    |
|weighted avg  |0.84         |0.83    |0.83       |640    |
|              |             |        |           |       |
|              |             |        |           |       |
|45m           |             |        |           |       |
|              |precision    |recall  |f1-score   |support|
|              |             |        |           |       |
|         low  |0.83         |0.76    |0.79       |217    |
|         mid  |0.63         |0.72    |0.67       |168    |
|        high  |0.95         |0.92    |0.93       |255    |
|              |             |        |           |       |
|    accuracy  |             |        |0.81       |640    |
|   macro avg  |0.80         |0.80    |0.80       |640    |
|weighted avg  |0.82         |0.81    |0.82       |640    |
|              |             |        |           |       |
|              |             |        |           |       |
|              |             |        |           |       |
|              |             |        |           |       |
|1h            |precision    |recall  |f1-score   |support|
|              |             |        |           |       |
|         low  |0.68         |0.83    |0.75       |222    |
|         mid  |0.57         |0.43    |0.49       |174    |
|        high  |0.95         |0.93    |0.94       |244    |
|              |             |        |           |       |
|    accuracy  |             |        |0.76       |640    |
|   macro avg  |0.73         |0.73    |0.72       |640    |
|weighted avg  |0.75         |0.76    |0.75       |640    |
|              |             |        |           |       |
|              |             |        |           |       |
|1h 15m        |precision    |recall  |f1-score   |support|
|              |             |        |           |       |
|         low  |0.68         |0.89    |0.77       |228    |
|         mid  |0.61         |0.40    |0.48       |178    |
|        high  |0.95         |0.90    |0.92       |234    |
|              |             |        |           |       |
|    accuracy  |             |        |0.76       |640    |
|   macro avg  |0.75         |0.73    |0.73       |640    |
|weighted avg  |0.76         |0.76    |0.75       |640    |
|              |             |        |           |       |
|              |             |        |           |       |
|1h 30m        |precision    |recall  |f1-score   |support|
|              |             |        |           |       |
|         low  |0.56         |1.00    |0.72       |232    |
|         mid  |0.00         |0.00    |0.00       |184    |
|        high  |0.92         |0.93    |0.92       |224    |
|              |             |        |           |       |
|    accuracy  |             |        |0.69       |640    |
|   macro avg  |0.49         |0.64    |0.55       |640    |
|weighted avg  |0.52         |0.69    |0.58       |640    |







