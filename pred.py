import cv2
from ultralytics import YOLO
import numpy as np
import base64
import time
from fastapi import FastAPI
from fastapi.responses import FileResponse
# Load the YOLOv8 model

model = YOLO('./runs/detect/train28/weights/best.pt')  # Use the YOLOv8 model, or load your custom model

# Load the shirt image
shirt_image = cv2.imread('./shirts/shirt.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available
lshoulder_image = cv2.imread('./shirts/lshoulder.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available
rshoulder_image = cv2.imread('./shirts/rshoulder.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

princess_shirt = cv2.imread('./shirts/princessshirt.png', cv2.IMREAD_UNCHANGED)
princessl = cv2.imread('./shirts/princessl.png', cv2.IMREAD_UNCHANGED)
princessr=cv2.imread('./shirts/princessr.png', cv2.IMREAD_UNCHANGED)

glasses=cv2.imread('./shirts/g.png', cv2.IMREAD_UNCHANGED)
helmet=cv2.imread('./shirts/helmet.png', cv2.IMREAD_UNCHANGED)
paths="./shirts/"

s3 = cv2.imread('./shirts/3s.png', cv2.IMREAD_UNCHANGED)
l3 = cv2.imread('./shirts/3r.png', cv2.IMREAD_UNCHANGED)
r3=cv2.imread('./shirts/3r.png', cv2.IMREAD_UNCHANGED)

s4 = cv2.imread('./shirts/4s.png', cv2.IMREAD_UNCHANGED)
l4 = cv2.imread('./shirts/4l.png', cv2.IMREAD_UNCHANGED)
r4=cv2.imread('./shirts/4r.png', cv2.IMREAD_UNCHANGED)

s5 = cv2.imread('./shirts/5s.png', cv2.IMREAD_UNCHANGED)
l5 = cv2.imread('./shirts/5l.png', cv2.IMREAD_UNCHANGED)
r5=cv2.imread('./shirts/5r.png', cv2.IMREAD_UNCHANGED)

s6 = cv2.imread('./shirts/6s.png', cv2.IMREAD_UNCHANGED)
l6 = cv2.imread('./shirts/6l.png', cv2.IMREAD_UNCHANGED)
r6=cv2.imread('./shirts/6r.png', cv2.IMREAD_UNCHANGED)

s7 = cv2.imread('./shirts/7s.png', cv2.IMREAD_UNCHANGED)
l7 = cv2.imread('./shirts/7l.png', cv2.IMREAD_UNCHANGED)
r7=cv2.imread('./shirts/7r.png', cv2.IMREAD_UNCHANGED)

g2 = cv2.imread('./shirts/g2.png', cv2.IMREAD_UNCHANGED)
g3 = cv2.imread('./shirts/g3.png', cv2.IMREAD_UNCHANGED)
g4=cv2.imread('./shirts/g4.png', cv2.IMREAD_UNCHANGED)
g5=cv2.imread('./shirts/g5.png', cv2.IMREAD_UNCHANGED)

listOutfits=[[shirt_image,lshoulder_image,rshoulder_image],[princess_shirt,princessl,princessr],[s3,l3,r3],[s4,l4,r4],[s5,l5,r5],[s6,l6,r6],[s7,l7,r7]]
glassesList = [glasses,g2,g3,g4,g5]
# Open the webcam (0 is the default camera)
 
cap = cv2.VideoCapture(0)
shirtx=1920
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()



start_time = time.time()
index=0
index2=0
while True:
    if time.time()-start_time > 3:
        start_time = time.time()
        index=(index+1)%7
        index2=(index+1)%5
        

    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Predict with YOLOv8 on the current frame
    results = model.predict(source=frame, save=False, save_txt=False, conf=0.75)  # Explicitly calling .predict()
    
    applied_shirt=listOutfits[index][0]
    applied_left = listOutfits[index][1]
    applied_right = listOutfits[index][2]
    # Extract and print the prediction results
    for result in results:
        boxes = (result.boxes.xyxy)  # Bounding box coordinates
        scores = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls   # Class labels
        
        print("here")
        print(result.boxes.xyxy)
        print("here2")
        # Loop through detected objects
        for box, score, label in zip(boxes, scores, labels):
            print(f"Box: {box}, Confidence: {score}, Class: {label}")
            print(label)
            # Assuming 'shirt' is detected as a certain class, e.g., label 0
            if int(label) == 1:  # Replace with the correct class index for the shirt
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
                shirtx=(x1+x2)//2
                # Draw the bounding box around the detected shirt
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                # Calculate the width and height of the bounding box
                box_width = x2 - x1
                box_height = y2 - y1

                # Resize the shirt image to fit the bounding box
                shirt_resized = cv2.resize(applied_shirt, (box_width, box_height))

                # Overlay the resized shirt image onto the original frame
                if shirt_resized.shape[2] == 4:  # If there is an alpha channel
                    alpha_shirt = shirt_resized[:, :, 3] / 255.0  # Alpha channel for transparency
                    alpha_frame = 1.0 - alpha_shirt

                    for c in range(0, 3):  # Iterate through color channels (BGR)
                        frame[y1:y2, x1:x2, c] = (alpha_shirt * shirt_resized[:, :, c] +
                                                alpha_frame * frame[y1:y2, x1:x2, c])
                else:
                    # If no alpha channel, just overlay it directly
                    frame[y1:y2, x1:x2] = shirt_resized
            elif int(label) == 2:  # Replace with the correct class index for the shirt
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
                newImage=applied_left
                if x1<shirtx:
                    newImage=applied_left
                else:
                    newImage=applied_right
                # Draw the bounding box around the detected shirt
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                # Calculate the width and height of the bounding box
                box_width = x2 - x1
                box_height = y2 - y1

                # Resize the shirt image to fit the bounding box
                shirt_resized = cv2.resize(newImage, (box_width, box_height))

                # Overlay the resized shirt image onto the original frame
                if shirt_resized.shape[2] == 4:  # If there is an alpha channel
                    alpha_shirt = shirt_resized[:, :, 3] / 255.0  # Alpha channel for transparency
                    alpha_frame = 1.0 - alpha_shirt

                    for c in range(0, 3):  # Iterate through color channels (BGR)
                        frame[y1:y2, x1:x2, c] = (alpha_shirt * shirt_resized[:, :, c] +
                                                alpha_frame * frame[y1:y2, x1:x2, c])
                else:
                    # If no alpha channel, just overlay it directly
                    frame[y1:y2, x1:x2] = shirt_resized
            elif int(label) == 0:  # Replace with the correct class index for the shirt
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
                newImage=glassesList[index2]
                
                # Draw the bounding box around the detected shirt
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                # Calculate the width and height of the bounding box
                box_width = x2 - x1
                box_height = y2 - y1

                # Resize the shirt image to fit the bounding box
                shirt_resized = cv2.resize(newImage, (box_width, box_height))

                # Overlay the resized shirt image onto the original frame
                if shirt_resized.shape[2] == 4:  # If there is an alpha channel
                    alpha_shirt = shirt_resized[:, :, 3] / 255.0  # Alpha channel for transparency
                    alpha_frame = 1.0 - alpha_shirt

                    for c in range(0, 3):  # Iterate through color channels (BGR)
                        frame[y1:y2, x1:x2, c] = (alpha_shirt * shirt_resized[:, :, c] +
                                                alpha_frame * frame[y1:y2, x1:x2, c])
                else:
                    # If no alpha channel, just overlay it directly
                    frame[y1:y2, x1:x2] = shirt_resized
            
    

    # Display the frame
    cv2.imwrite("./RowdyHack 2024/image.jpg",frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
