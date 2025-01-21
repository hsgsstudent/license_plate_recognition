import src.data_utils as utils
import cv2
import numpy as np
from ultralytics import YOLO



class detectNumberPlate(object):
    def __init__(self, classes_path, model_path, threshold=0.5):

        self.labels = utils.get_labels(classes_path)
        self.threshold = threshold

        # Load model
        #load_model
        self.model = YOLO(model_path)

    def detect(self, image):
        # Convert image to RGB (YOLO expects RGB images)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = self.model.predict(source=rgb_image)

        # Process results
        coordinates = []
        for result in results:
            for box in result.boxes.data:
                x_min, y_min, x_max, y_max, confidence, class_id = box.tolist()

                if confidence > self.threshold:
                    width = x_max - x_min
                    height = y_max - y_min
                    coordinates.append((int(x_min), int(y_min), int(width), int(height)))
                    
        return coordinates


