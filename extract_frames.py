import cv2
import os

# Directories
video_path = 'ball_tracking_video.mp4'
training_output_folder = 'dataset/Training/images'
validation_output_folder = 'dataset/Validation/images'
# os.makedirs(training_output_folder, exist_ok=True)
os.makedirs(validation_output_folder, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize
start_frame = 5                                                        # Set 0 for training ; 5 for Validation
interval = 10
frame = 0
count = 0

# Loop through the video and save every 10th frame starting from the 5th frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Check if the current frame is at the specified interval
    if frame >= start_frame and (frame - start_frame) % interval == 0:
        # Save the frame with zero-padded naming convention
        # frame_name = os.path.join(training_output_folder, f'frame_{frame:04}.jpg')
        frame_name = os.path.join(validation_output_folder, f'frame_{frame:04}.jpg')
        cv2.imwrite(frame_name, frame)
        count += 1

    frame += 1

cap.release()

# print(f"Extracted {count} training frames and saved them in the '{training_output_folder}' folder.")
print(f"Extracted {count} validation frames and saved them in the '{validation_output_folder}' folder.")
