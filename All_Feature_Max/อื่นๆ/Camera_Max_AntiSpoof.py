import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล
model = YOLO("Models/latestversion.pt")

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame")
        break

    # รันโมเดล YOLO บนเฟรม
    results = model.predict(source=frame, conf=0.5, show=False)

    # วาด Bounding Boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # แสดงภาพ
    cv2.imshow("Face Anti-Spoofing", frame)

    # กด 'q' เพื่อปิด
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
