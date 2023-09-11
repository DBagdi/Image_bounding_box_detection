import os
import cv2

from ultralytics import YOLO

# Path to the input image (image.png)
image_path = 'images/val/val2.jpg'

# Load the input image
frame = cv2.imread(image_path)

# Get image dimensions
H, W, _ = frame.shape

# Initialize Ultralytics YOLO model
model_path = 'C:\\Users\\91749\\runs\\detect\\train2\\weights\\last.pt'
model = YOLO(model_path)  # Load a custom model


# Confidence threshold
threshold = 0.5

# Run object detection on the input image
results = model(frame)[0]

# Iterate through detected objects and draw bounding boxes
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Display or save the resulting image
output_image_path = 'images/output/output_val2.jpg'
cv2.imwrite(output_image_path, frame)

# Release resources
cv2.destroyAllWindows()

print(f"Detection results saved to {output_image_path}")