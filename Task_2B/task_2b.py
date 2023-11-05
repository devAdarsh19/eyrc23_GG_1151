'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2B of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			GG_1151
# Author List:		Adarsh Vinod, Madan Maskara, Pooja Vishwakarma, Aayush Kumar
# Filename:			task_2b.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os


# Additional Imports
import random
import torch
from PIL import Image
import torchvision
import PIL
import torchvision.transforms as transforms
import glob
from torch.utils.data import DataLoader
import pathlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets,transforms,models
import torch.nn as nn
import torch.optim as optim

# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
detected_list = []
numbering_list = []
img_name_list = []

# Declaring Variables
# answer=None
classess=["combat","destroyedbuilding","fire","humanitarianaid","militaryvehicles"]

# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
###################################################################################################
###################################################################################################
''' 
	Purpose:
	---
	This function will load your trained model and classify the event from an image which is 
    sent as an input.
	
	Input Arguments:
	---
	`image`: Image path sent by input file 	
	
	Returns:
	---
	`event` : [ String ]
						  Detected event is returned in the form of a string

	Example call:
	---
	event = classify_event(image_path)
	'''

mean = [0.4873, 0.4838, 0.4746]
std = [0.2585, 0.2584, 0.2712]

image_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])
path='C:/Users/AnshulKumar/Desktop/war/drive-download-20231023T114515Z-001'
model=torch.load(path+'/weight.pth')
def set_device():
  if torch.cuda.is_available():
     dev = "cuda:0"
  else:
     dev = "cpu"
  return torch.device(dev)

device = set_device()

def classify(model, image_transforms, image_path, classes):
    model.eval()
    model = model.to(device)
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image.to(device))

    _, predicted = torch.max(output.data, 1)
    # print(predicted[0].item())
    answer=classess[int(predicted[0].item())]
    # answer=classess[0]
    # print(classes[predicted.item()])
    return answer

def classify_event(image):
    event=classify(model, image_transforms, image,5)
    # event = answer
    return event

# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''


###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(img_name_list):
    for img_index in range(len(img_name_list)):
        img = "events/" + str(img_name_list[img_index]) + ".jpeg"
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    shutil.rmtree('events')
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)
    img_names = open("image_names.txt", "r")
    img_name_str = img_names.read()

    img_name_list = ast.literal_eval(img_name_str)
    return img_name_list
    
def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)

###################################################################################################
def main():
    ##### Input #####
    img_name_list = input_function()
    #################

    ##### Process #####
    detected_list = classification(img_name_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('events'):
            shutil.rmtree('events')
        sys.exit()
###################################################################################################
###################################################################################################
