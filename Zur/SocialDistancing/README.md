Social Distancing Detector

Welcome to my social distancing detector!

What it does:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Takes video input  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Detects people and calculates distance between each pair of people  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. Indicates if any person is standing less than 6 feet away from another person  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. Outputs txt file with data that is collected over time  

How to use:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Download YOLO object detector by running (./download_model.sh)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Make sure all dependencies are installed (pip install -r requirements.txt)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. Usage: python sdd.py --input (path to video)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If no input path to video is provided, webcam is used  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Videos folder provided with sample pedestrian videos  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. Other arguments with defaults that can be adjusted:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a. --confidence (float): confidence threshold for yolo detection (default=0.5)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b. --threshold (float): non-maximum suppression algorithm threshold (default=0.3)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c. --disappeared (int): number of frames that a registered object ID is not detected before it is de-registered (default=10)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;d. --frames (int): number of frames between data outputs to text file (default=20)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5. Follow instructions on the image that pops up to input the six mouse points for the region of interest and the six-foot approximation  

Read more about the project and how it works: https://sagecontinuum.org/science/social-distancing/
