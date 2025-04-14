import cv2
import pandas as pd
from ultralytics import YOLO

'''''''''Training Model''''''
# Load the pretrained YOLOv8 model
model = YOLO('yolov8m.pt')  

# Training the YOLO model 
model.train(
    data='data.yaml',
    epochs=50,             # Increase epochs
    imgsz=960,             # Use a larger image size for detailed objects
    batch=8,               # Adjust batch size based on memory
    name='soccer_ball_test'
)

'''

model = YOLO("runs/detect/soccer_ball_test/weights/best.pt")                       # Loading the trained model

# Directories
input_video_path = "ball_tracking_video.mp4"
output_video_path = "output/output_tracked_video.mp4"
csv_output_path = "output/output_tracking_data.csv"

# Initialize video capture and writer, frame number and the csv list which tracks the soccer ball
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
list_csv = []
frame_number = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, save=False, show=False)                         # Running YOLOv8 inference

    # Detecting the soccer ball
    if results and hasattr(results[0], 'boxes') and results[0].boxes:
        for i, box in enumerate(results[0].boxes.data):
            if box.shape[0] >= 4:
                x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])     # Extract bounding box coordinates

                # Center and size of the bounding boxes
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                x_size = x_max - x_min
                y_size = y_max - y_min

                frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)        # Drawing the bounding box on the frame
                
                detection_data = [frame_number, x_center, y_center, x_size, y_size]
                list_csv.append(detection_data)                                                     # Append the bounding frame and bounding box data to the list_csv

    out.write(frame)
    frame_number += 1
cap.release()
out.release()

# Save the collected data to a CSV file
if list_csv:
    df = pd.DataFrame(list_csv, columns=["frame_number", "x_center", "y_center", "x_size", "y_size"])
    df.to_csv(csv_output_path, index=False)
    print(f"Data saved to CSV successfully at {csv_output_path}")
else:
    print("No data to save to CSV.")
