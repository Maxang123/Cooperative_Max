import cv2
import insightface
from insightface.app import FaceAnalysis

# เตรียมโมเดลสำหรับตรวจจับใบหน้า
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ใช้ GPU (หรือเปลี่ยนเป็น -1 สำหรับ CPU)

def process_face_blur(frame):
    """
    เบลอใบหน้าในเฟรมด้วย GaussianBlur
    """
    faces = face_app.get(frame)
    if faces:
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            # ตรวจสอบขอบเขตให้ถูกต้อง
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])
            face_region = frame[y1:y2, x1:x2]
            blurred_region = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y1:y2, x1:x2] = blurred_region
    return frame

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ประมวลผลเบลอใบหน้าจากเฟรม
    frame = process_face_blur(frame)
    
    # แสดงผลแบบ real-time
    cv2.imshow("Real-Time Face Blur", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
