# Rick's Repo Home Page

<br>

## AMS Poster
### https://annual.ametsoc.org/index.cfm/2021/program-events/conferences-and-symposia/20th-conference-on-artificial-intelligence-for-environmental-science/
### Abstract due August 20th (limit 1000 words)
### Meeting, taking place from 10-14 January 2021 in New Orleans.
* The Standard Poster Display is a 4' x 8' (1.2m x 2.4m) double-sided Velcro covered panel. These panels have a horizontal/landscape orientation. This varies per conference. Please check the conference specific poster information.
* You can mount your poster with AMS supplied pushpins or Velcro.
* Place the title at the top center of the poster.
* Use 72 point letters.
* Avoid hand drawn materials.
* Keep it simple.
* Do not post the pages of your paper.
* Make sure it has a logical flow.
* Use color, graphics, charts, and photos.
* Key points should be in large type.
* Do not try to tell the entire story on the panel.
* Save some key points for the one-on-one discussion.
* Avoid the following color combinations: Green&Red; Green&Brown; Blue&Purple; Green&Blue; Light Green&Yellow; Blue&Grey; Green&Grey; Green&Black

## To do list :)
01. [ ] Update paper (morning)
02. [ ] Work on getting tensorflow/cuda working on node (afternoon)
03. [ ] Get data from lambda for E2 and E3 (morning)
04. [ ] Build parser for video rain data (Afternoon) 

<br>

## WeatherNet_v2
* This is the main directory of the project! All of the codes needed to reconstruct the experiment can be found in here!

<br>

## ImgSeqLoader [ This goes to WeatherNet_v1 ]
* This directory contains code for the data loader I made for my model that forecasts solar radiation amounts. This data loader feeds in three sets of time-sequenced images from three different cameras. The data loader navigates the folder structure (as shown below) and grabs the correct called images and deletes from memory the old images from memory.

![tree](Images/DataLoaderDirPic.png)                                                                                          
                                                                                         
In the repo, you will find an example .ipynb notebook showing how the data loader and how it fetches images. There are limited examples of how to build a data loader on the internet with multiple inputs and also for sequence-based images, so this may be a useful resource for constructing your own data loader for image sequences (video). 

<br>

## WeatherNet_v1
* This directory contains code for the WeatherNetV1. WeatherNetV1 would not learn to be able to predict out of sample. The following reason are why I believe it did not work.
1. The model did not know the current value of solar energy.
2. The size of 6 images for a single sample was not ideal since again, each sample did not have a corresponding intial condition. 
3. My belief is that the model with be highly influenced by the brightness of the images and cloud coverage (which in hand will also affect brightness)

![tree](Images/WeatherNet_v1_pic.png) 

<br>

## LSTM Solar Forecast [This is the model from experiment two in the paper]
* For my experiment, one of the test consist of using a LSTM to forecast solar radiation. This is used as a baseline to compare the performance of WeatherNet.
Inside this directory, is an array of files however, for the finshed and documented experiment refer to the folder LSTM_Weather_Data_Experiment.

![tree](Images/LSTM_Weather_Model_netron.png) 

## ProjectUpdates
* This repo contains slides from my weekly update meetings.

## log
* Inside this directory, you will find my daily schedules, which document what I worked on each day during summer.

<br>

## DataPrep
* DataPrep is an assortment of python scripts used to build the dataset. This folder is not organized for recreating the dataset but merely was a dump for my work while making the dataset. THIS FOLDER'S CODE DOES NOT NEED TO BE READ IN ORDER TO UNDERSTAND THE PROJECT (save yourself the time).

