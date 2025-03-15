import numpy as np
import tensorflow as tf
import cv2
import time
import os

model_path = './pallet_detectV1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_height = input_details[0]['shape'][1]  # 640
image_width = input_details[0]['shape'][2]   # 640

# Initialize webcam
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise ValueError("Unable to open webcam")

# Threshold for detection confidence
threshold = 0.3
inference_active = False  
image_counter = 0
max_images = 100  # Maximum number of images to save

class_id_pallet = 0  # Change this to the correct class ID for wood pallets

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to match the input tensor size
    frame_resized = cv2.resize(frame, (image_width, image_height))

    # Draw a centered rectangle box based on the original frame dimensions
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    rect_width, rect_height = 500, 100
    cv2.rectangle(frame, 
                  (center_x - rect_width // 2, center_y - rect_height // 2), 
                  (center_x + rect_width // 2, center_y + rect_height // 2), 
                  (255, 0, 0), 2)

    if inference_active:
        # Preprocess the frame
        frame_np = np.array(frame_resized, dtype=np.float32) / 255.0
        frame_np = frame_np[np.newaxis, :]

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], frame_np)

        # Perform inference
        start = time.time()
        interpreter.invoke()
        print(f'Inference time: {time.time() - start:.2f}s')

        # Get the output tensor
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0]
        output = output.T

        if output.size > 0:
            boxes_xywh = output[..., :4]
            scores = np.max(output[..., 4:], axis=1)
            classes = np.argmax(output[..., 4:], axis=1)

            for box, score, cls in zip(boxes_xywh, scores, classes):
                if score >= threshold and cls == class_id_pallet:  # Check for wood pallet class
                    x_center, y_center, width, height = box
                    x1 = int((x_center - width / 2) * image_width)
                    y1 = int((y_center - height / 2) * image_height)
                    x2 = int((x_center + width / 2) * image_width)
                    y2 = int((y_center + height / 2) * image_height)

                    # Draw bounding box on the original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Class: {cls}, Score: {score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Crop the detected bounding box region
                    cropped_frame = frame[y1:y2, x1:x2]

                    # Save the cropped frame as a JPEG file if within limit
                    if image_counter < max_images:
                        cv2.imwrite(f'detected_frame_{image_counter}.jpg', cropped_frame)
                        image_counter += 1  # Increment counter for the next image
                    else:
                        print("Maximum number of images reached. Not saving more.")

    cv2.imshow("Webcam Inference", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        inference_active = not inference_active
        state = "ON" if inference_active else "OFF"
        print(f"Inference is now {state}")

    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()