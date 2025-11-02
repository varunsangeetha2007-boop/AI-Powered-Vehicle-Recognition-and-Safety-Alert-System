from ultralytics import YOLO
import cv2
import pygame
pygame.mixer.init()
model = YOLO('yolov8n.pt')