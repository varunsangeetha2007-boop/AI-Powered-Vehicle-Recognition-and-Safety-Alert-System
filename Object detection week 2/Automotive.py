from ultralytics import YOLO
import cv2
import pygame

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")  # use your sound file

# Load YOLO model
model = YOLO('yolov8n.pt')

# 👉 Replace with your uploaded image path
img_path = "car.jpg"  

# Load image
frame = cv2.imread(img_path)

# Detect objects
results = model(frame)
annotated_frame = results[0].plot()

# Check for vehicles
detected_vehicle = False
for box in results[0].boxes:
    cls = int(box.cls[0])
    label = model.names[cls]
    if label in ["car", "bus", "truck", "motorbike"]:
        detected_vehicle = True
        cv2.putText(annotated_frame, "⚠ Vehicle Detected!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Play sound if detected
if detected_vehicle:
    alert_sound.play()

# Show output
cv2.imshow("Smart Vehicle Object Detection (Image)", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
