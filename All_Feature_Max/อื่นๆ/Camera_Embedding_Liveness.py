import os
import cv2
import numpy as np
import pickle
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import mediapipe as mp
import time

# ============================
# 1. เตรียมโมเดลและตัวแปรพื้นฐาน
# ============================

# Initialize InsightFace FaceAnalysis (ArcFace)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))  # ใช้ GPU: ctx_id=0, CPU: ctx_id=-1

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# สำหรับวาด landmark (optional)
mp_drawing = mp.solutions.drawing_utils

# ตัวแปรสำหรับเก็บสถานะพฤติกรรม
triggered_behavior = None       # เช่น "Blink 1" หรือ "Identity verification success"
behavior_timestamp = 0          # เวลาเมื่อพฤติกรรมถูก trigger

# ตัวแปรสำหรับตรวจจับการกระพริบตา
blink_count = 0
blink_in_progress = False
verification_success = False
BLINK_THRESHOLD = 5  # threshold สำหรับตรวจจับการกระพริบตา (อาจต้องปรับ)

# ตัวแปรสำหรับตรวจจับการหันหน้า (head turn)
head_turn_behavior = None       # เช่น "Turn Left" หรือ "Turn Right"
head_turn_timestamp = 0          # เวลาเมื่อพฤติกรรมถูกตรวจจับ

# ============================
# 2. โหลด Embeddings และฟังก์ชันสำหรับ Face Recognition
# ============================

def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

data = load_embeddings('embeddings_Best.pkl')

def extract_embeddings(image):
    faces = app.get(image)
    embeddings = []
    bboxes = []
    for face in faces:
        embeddings.append(face.normed_embedding)
        bboxes.append(face.bbox)  # พิกัด bounding box
    return embeddings, bboxes

def match_face(new_face_embedding, data):
    if new_face_embedding.size == 0:
        return None, 0
    best_match = None
    best_score = -1
    new_face_embedding = np.array(new_face_embedding).reshape(1, -1)
    for person, embeddings in data.items():
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        scores = cosine_similarity(new_face_embedding, embeddings)
        max_score = max(scores[0])
        if max_score > best_score:
            best_match = person
            best_score = max_score
    return best_match, best_score

# ============================
# 3. เริ่มการอ่านภาพจากกล้องและประมวลผล
# ============================

cap = cv2.VideoCapture(0)

# กำหนด ROI สำหรับให้ผู้ใช้วางใบหน้า (ปรับค่าตามความเหมาะสม)
roi_x1, roi_y1 = 150, 100
roi_x2, roi_y2 = 490, 380

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถเปิดกล้องได้")
        break
    current_time = time.time()

    # วาดกรอบ ROI บนเฟรม
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    
    # --- Face Recognition ด้วย InsightFace ---
    embeddings, bboxes = extract_embeddings(frame)
    if embeddings:
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            
            # ตรวจสอบว่าใบหน้ามีตำแหน่งอยู่ภายใน ROI หรือไม่
            if not (roi_x1 <= face_center_x <= roi_x2 and roi_y1 <= face_center_y <= roi_y2):
                cv2.putText(frame, "Please align your face in the frame", (0, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue  # ข้ามใบหน้าที่อยู่นอก ROI
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (152, 0, 255), 2)
            result, score = match_face(embeddings[i], data)
            if result:
                cv2.putText(frame, f"{result} ({score:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (152, 0, 255), 2)
            else:
                cv2.putText(frame, "No match found", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (152, 0, 255), 2)

    # แปลงภาพเป็น RGB สำหรับ Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_mesh_results = face_mesh.process(rgb_frame)

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            h, w, _ = frame.shape
            # คำนวณ bounding box จาก landmark ทั้งหมด
            x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            face_box_width = max_x - min_x
            face_box_height = max_y - min_y
            face_center_x = (min_x + max_x) // 2
            face_center_y = (min_y + max_y) // 2

            # ตรวจสอบว่าใบหน้ามีขนาดพอและอยู่ใน ROI (ถ้าไม่ ให้แจ้งให้ผู้ใช้ปรับตำแหน่ง)
            if face_box_width < 150 or face_box_height < 150 or not (roi_x1 <= face_center_x <= roi_x2 and roi_y1 <= face_center_y <= roi_y2):
                cv2.putText(frame, "Please move closer and align face", (0,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
            
            # Optional: วาด bounding box จาก landmarks
            # cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)

            # --- ตรวจจับพฤติกรรมบนใบหน้าด้วย Mediapipe Face Mesh ---

            """
            ตรวจจับการยิ้ม: โดยใช้ landmark ที่ตำแหน่งมุมซ้ายและขวาปาก (landmark 61 และ 291) 
            คำนวณความกว้างของปากและเปรียบเทียบกับความกว้างของใบหน้าที่ประเมินจากแก้มทั้งสองข้าง (landmark 234 และ 454) 
            หากอัตราส่วนเกิน 0.5 จะถือว่ามีรอยยิ้มและแสดงข้อความ “Smile (¯▽¯)”  
            """
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            pt_left_mouth = np.array([left_mouth.x * w, left_mouth.y * h])
            pt_right_mouth = np.array([right_mouth.x * w, right_mouth.y * h])
            mouth_width = np.linalg.norm(pt_left_mouth - pt_right_mouth)
            # ประมาณความกว้างของใบหน้าโดยใช้ landmark 234 (แก้มซ้าย) และ 454 (แก้มขวา)
            left_face = face_landmarks.landmark[234]
            right_face = face_landmarks.landmark[454]
            pt_left_face = np.array([left_face.x * w, left_face.y * h])
            pt_right_face = np.array([right_face.x * w, right_face.y * h])
            face_width = np.linalg.norm(pt_left_face - pt_right_face)
            if face_width > 0:
                smile_ratio = mouth_width / face_width
                if smile_ratio > 0.5:
                    triggered_behavior = "Smile (¯▽¯)"
                    behavior_timestamp = current_time

            # **ตรวจจับการกระพริบตา (Blink) สำหรับยืนยันตัวตน**
            """
            ตรวจจับการกระพริบตา (สำหรับยืนยันตัวตน): ใช้ landmark ของดวงตาซ้าย (top: landmark 159, bottom: landmark 145) 
            คำนวณระยะห่างระหว่างสองจุด หากระยะห่างน้อยกว่า BLINK_THRESHOLD ถือว่ากระพริบตาและนับ blink_count เมื่อกระพริบตาครบ 3 ครั้ง 
            จะถือว่าการยืนยันตัวตนสำเร็จและแสดงข้อความ “Identity verification success (O_o)” 
            """
            if not verification_success:
                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]
                pt_top = np.array([left_eye_top.x * w, left_eye_top.y * h])
                pt_bottom = np.array([left_eye_bottom.x * w, left_eye_bottom.y * h])
                eye_distance = np.linalg.norm(pt_top - pt_bottom)
                
                if eye_distance < BLINK_THRESHOLD:
                    if not blink_in_progress:
                        blink_in_progress = True
                        blink_count += 1
                        triggered_behavior = f"Blink {blink_count}! (>_<)"
                        behavior_timestamp = current_time
                        if blink_count >= 3:
                            verification_success = True
                            triggered_behavior = "Identity verification success (O_o)"
                            behavior_timestamp = current_time
                else:
                    blink_in_progress = False

            # **ตรวจจับการหันหน้าซ้าย/ขวา (Head Turn)**
            """
            ตรวจจับการหันหน้า: โดยใช้ landmark ของจมูก (landmark 1) และดวงตาซ้าย-ขวา (landmark 33 และ 263) 
            คำนวณความแตกต่างของตำแหน่งในแกน x ถ้าน้ำหนักของใบหน้าหันไปด้านขวาหรือซ้ายเกินค่าที่กำหนด (15 พิกเซล) 
            จะแสดงข้อความ “Turn Right-->” หรือ “Turn Left<--” 
            """
            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_x = nose.x * w
            left_eye_x = left_eye.x * w
            right_eye_x = right_eye.x * w
            eyes_mid_x = (left_eye_x + right_eye_x) / 2
            if (eyes_mid_x - nose_x) > 15:
                head_turn_behavior = "Turn Right-->"
                head_turn_timestamp = current_time
            elif (nose_x - eyes_mid_x) > 15:
                head_turn_behavior = "Turn Left<--"
                head_turn_timestamp = current_time

            # วาด landmark บนใบหน้า (optional)
            # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            #                           landmark_drawing_spec=None,
            #                           connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
    else:
        # เมื่อไม่พบใบหน้าในเฟรม ให้รีเซ็ตค่าพฤติกรรม
        blink_count = 0
        blink_in_progress = False
        verification_success = False
        triggered_behavior = None
        head_turn_behavior = None
        behavior_timestamp = 0
        head_turn_timestamp = 0

    # --- แสดงข้อความบนเฟรม ---
    if not verification_success:
        cv2.putText(frame, "Please blink 3 times slowly", (0, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    if triggered_behavior is not None and (current_time - behavior_timestamp) < 1:
        cv2.putText(frame, triggered_behavior, (0, 400), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)
    if head_turn_behavior is not None and (current_time - head_turn_timestamp) < 1:
        cv2.putText(frame, head_turn_behavior, (0, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)

    cv2.imshow("Face Recognition with Blink Verification😎", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
