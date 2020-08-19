Social Distancing Detector

Welcome to my social distancing detector!

What it does:  
	1. Takes video input  
	2. Detects people and calculates distance between each pair of people  
	3. Indicates if any person is standing less than 6 feet away from another person  
	4. Outputs txt file with data that is collected over time  

How to use:  
	1. Download YOLO object detector by running (./download_model.sh)  
	2. Make sure all dependencies are installed (pip install -r requirements.txt)  
	3. Usage: python sdd.py --input (path to video)  
		example: python sdd.py --input videos/test_video2.mp4  
		if no input path to video is provided, webcam is used  
		videos folder provided with some sample pedestrian videos  
	4. Other arguments with defaults that can be adjusted:  
		a. --confidence (float): confidence threshold for yolo detection (default=0.5)  
		b. --threshold (float): non-maximum suppression algorithm threshold (default=0.3)  
		c. --frames (int): number of frames between data outputs to text file (default=20)  
        5. Follow instructions on the image that pops up  

Read more about the project and how it works: https://sagecontinuum.org/science/social-distancing/
