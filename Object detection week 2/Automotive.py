from ultralytics import YOLO
import cv2
import pygame

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav") 

model = YOLO('yolov8n.pt')

img_path = "car.jpg"  

frame = cv2.imread(img_path)

results = model(frame)
annotated_frame = results[0].plot()

detected_vehicle = False
for box in results[0].boxes:
    cls = int(box.cls[0])
    label = model.names[cls]
    if label in ["car", "bus", "truck", "motorbike"]:
        detected_vehicle = True
        cv2.putText(annotated_frame, "Vehicle Detected!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

if detected_vehicle:
    alert_sound.play()

cv2.imshow("Smart Vehicle Object Detection (Image)", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
