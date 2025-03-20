import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors
model_path = './pallet_detectV1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image: resize and normalize if necessary
    input_shape = input_details[0]['shape']
    input_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_frame)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assuming output_data is a 2D array with shape [num_detections, 6]
    for detection in output_data:
        # Check if the detection is not empty and has the expected number of elements
        if detection.size >= 6 and detection[4] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = detection[0], detection[1], detection[2], detection[3]
            # Convert to integer pixel values
            (left, right, top, bottom) = (int(xmin * frame.shape[1]), int(xmax * frame.shape[1]),
                                           int(ymin * frame.shape[0]), int(ymax * frame.shape[0]))

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Optionally, put class label and confidence on the bounding box
            label = f"Class: {int(detection[5])}, Conf: {detection[4]:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()