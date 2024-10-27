import cv2
from ultralytics import YOLO
import numpy as np

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
listOutfits=[[shirt_image,lshoulder_image,rshoulder_image],[princess_shirt,princessl,princessr]]
# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)
shirtx=1920
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Predict with YOLOv8 on the current frame
    results = model.predict(source=frame, save=False, save_txt=False, conf=0.75)  # Explicitly calling .predict()
    index=0
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
                newImage=glasses
                
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
    cv2.imshow('YOLOv8 Webcam', frame)  # Use the updated frame with the overlay

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
