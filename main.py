import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from coco_labels import COCO_LABELS

print("Carregando o modelo pré-treinado (SSD MobileNet V2)...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Modelo carregado com sucesso.")

def draw_boxes(frame, boxes, scores, classes, threshold=0.5):
    height, width, _ = frame.shape
    for i in range(boxes.shape[0]):
        if scores[i] < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        start_point = (int(xmin * width), int(ymin * height))
        end_point = (int(xmax * width), int(ymax * height))
        
        class_id = int(classes[i])
        label = COCO_LABELS.get(class_id, f"ID: {class_id}")
        
        text = f"{label}: {scores[i]:.2f}"
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(frame, text, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)

print("Iniciando a detecção de objetos em tempo real. Pressione 'q' para sair.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]  

    
    detections = detector(input_tensor)
    
    
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)

    draw_boxes(frame, detection_boxes, detection_scores, detection_classes, threshold=0.5)

    cv2.imshow("Detecção de Objetos em Tempo Real", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
