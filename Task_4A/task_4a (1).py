import cv2 as cv
import os
from ultralytics import YOLO
import keyboard

# Team ID:           [ Team-ID ]
# Author List:       [ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:          task_4a.py

####################### IMPORT MODULES #######################

##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

def capture_and_process_webcam():
    # Open a video capture object for the external webcam (try different indices)
    cap = cv.VideoCapture(0)
    # cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 800)
    # cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 600)
    while(True):
        ret, arena = cap.read()
        
        if cv.waitKey(1) and 0xFF == ord('q'):
            break
        arena_g = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(arena_g, 217, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ret, thresh = cv.threshold(arena_g, 217, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(arena_g, contours, -1, (0,255,0), 2)
        event_list = []
        for i,contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            
            area = cv.contourArea(contour)
            if area > 1000 and area < 2000:
                event = arena[y:y+h, x:x+w]
                event_list.append(event)
                arena = cv.rectangle(arena, (x,y), (x+w, y+h), (0,255,0), 2)
                cv.putText(arena, 'event', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv.resize(arena,(800,600))
                cv.imshow('are',arena)
                
               
                
        if keyboard.is_pressed('c'):
                # Save the current frame to a file (you can customize the filename)
            cv.imwrite('captured_frame26.jpg', arena)
                #print("Frame captured!")
            #cv.destroyAllWindows()
            break
            
    cap.release()
    cv.destroyAllWindows()
    # Load the saved image
    arena = cv.imread('captured_frame26.jpg')

    # Convert the image to grayscale
    arena_g = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    ret, thresh = cv.threshold(arena_g, 217, 255, cv.THRESH_BINARY)

    # Find contours in the thresholded image
    #contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(arena, contours, -1, (0,255,0), 1)
    # cv.imshow(arena)

    # Create a folder to store images
    output_folder = 'final_run_images208'
    os.makedirs(output_folder, exist_ok=True)

    event_images_list = []
    j = 1
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)

        if 1000 < area < 2000:
            event = arena[y:y+h, x:x+w]
            event_images_list.append(event)
            print(len(event_images_list))

            # Save the event image to the output folder
            filename = os.path.join(output_folder, f'{j}.jpg')
            j += 1
            cv.imwrite(filename, event)

            cv.rectangle(arena, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(arena, 'event', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



    # Show the final image with contours
    cv.imshow('arena', arena)
    cv.waitKey(0)
    cv.destroyAllWindows()

def predict_event_classes(model, images_folder, classes):
    results = model(images_folder, imgsz=32)

    identified_labels = {}

    for result in results:
        extract = []
        lists = list(result.probs.data)
        for l in lists:
            extract.append(round(l.item(), 2))
        max_acc = max(result.probs.data)
        max_acc = round(max_acc.item(), 2)
        max_ind = extract.index(max_acc)

        # Store the predicted label in the identified_labels dictionary
        identified_labels[chr(ord('A') + len(identified_labels))] = str(classes[max_ind])

    return identified_labels

##############################################################

def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable

    Returns:
    ---
    identified_labels : { dictionary }
        dictionary containing the labels of the events detected
    """
    identified_labels = {}

    # Process webcam and save contours
    capture_and_process_webcam()

    # YOLO Model
    classes = ['combat', 'destroyedbuilding', 'fire', 'humanitarianaid', 'militaryvehicles']
    yolo_model = YOLO("./best (12).pt")





    # Predict event classes
    identified_labels = predict_event_classes(yolo_model, 'final_run_images208', classes)

    return identified_labels

############### Main Function #################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)