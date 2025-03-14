import os
import cv2
import numpy as np
import pickle
import time
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp

# ============================
# 1. เตรียมโมเดลและตัวแปรพื้นฐาน
# ============================

# Initialize InsightFace FaceAnalysis (ArcFace)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))  # ใช้ GPU: ctx_id=0, CPU: ctx_id=-1

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize Mediapipe Hands (optional สำหรับตรวจจับมือ)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# สำหรับวาด landmark (ทั้ง Face Mesh และ Hands)
mp_drawing = mp.solutions.drawing_utils

# ตัวแปรสำหรับเก็บสถานะพฤติกรรม
triggered_behavior = None       # เช่น "Blink 1", "Smile (¯▽¯)" หรือ "Identity verification success (O_o)"
behavior_timestamp = 0          # เวลาเมื่อพฤติกรรมถูก trigger

# สำหรับตรวจจับการกระพริบตา
blink_count = 0
blink_in_progress = False
verification_success = False
BLINK_THRESHOLD = 5  # ค่าที่ใช้เปรียบเทียบระยะห่างในหน่วยพิกเซล (อาจปรับ)

# สำหรับตรวจจับการหันหน้า (head turn)
head_turn_behavior = None       # "Turn Left<--" หรือ "Turn Right-->"
head_turn_timestamp = 0          # เวลาเมื่อตรวจพบ head turn

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
# 3. ฟังก์ชันสำหรับ Head Pose Estimation (ใช้ cv2.solvePnP)
# ============================

def estimate_head_pose(landmarks, w, h):
    # กำหนด 2D image points จาก Mediapipe landmarks (6 จุดสำคัญ)
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h),    # Chin
        (landmarks[33].x * w, landmarks[33].y * h),      # Left eye left corner
        (landmarks[263].x * w, landmarks[263].y * h),    # Right eye right corner
        (landmarks[61].x * w, landmarks[61].y * h),      # Left Mouth corner
        (landmarks[291].x * w, landmarks[291].y * h)     # Right Mouth corner
    ], dtype="double")
    
    # กำหนด 3D model points (ค่าประมาณในหน่วย mm)
    model_points = np.array([
        [0.0, 0.0, 0.0],             # Nose tip
        [0.0, -63.6, -12.5],         # Chin
        [-43.3, 32.7, -26.0],        # Left eye left corner
        [43.3, 32.7, -26.0],         # Right eye right corner
        [-28.9, -28.9, -24.1],       # Left Mouth corner
        [28.9, -28.9, -24.1]         # Right Mouth corner
    ])
    
    # ค่าพารามิเตอร์ของกล้อง (สมมติ focal length = ความกว้างของภาพ)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    
    dist_coeffs = np.zeros((4, 1))  # สมมติว่าไม่มี lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None, None
    
    # กำหนดแกน 3 มิติสำหรับวาด (ปรับความยาวแกนตามที่ต้องการ)
    axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
    imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    # คำนวณ Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    # Euler angles: [pitch, yaw, roll] (องศา)
    pitch, yaw, roll = euler_angles.flatten()
    
    return imgpts, (pitch, yaw, roll), rotation_vector, translation_vector

# ============================
# 4. เริ่มอ่านภาพจากกล้องและประมวลผล
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

    # พลิกภาพให้เหมือนกระจก
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # วาดกรอบ ROI บนเฟรม
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    # --- Face Recognition ด้วย InsightFace ---
    embeddings, bboxes = extract_embeddings(frame)
    if embeddings:
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2

            # ตรวจสอบว่าใบหน้าตรงกับ ROI หรือไม่
            if not (roi_x1 <= face_center_x <= roi_x2 and roi_y1 <= face_center_y <= roi_y2):
                cv2.putText(frame, "Please align your face in the frame", (0, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue

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
    hands_results = hands.process(rgb_frame)

    # --- Process Face Mesh (ตรวจจับใบหน้าและพฤติกรรม) ---
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            # คำนวณ bounding box จาก landmarks ทั้งหมด
            x_coords = [lm.x * w for lm in face_landmarks.landmark]
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            face_box_width = max_x - min_x
            face_box_height = max_y - min_y
            face_center_x = (min_x + max_x) // 2
            face_center_y = (min_y + max_y) // 2

            # ตรวจสอบขนาดใบหน้าและตำแหน่งใน ROI
            if face_box_width < 150 or face_box_height < 150 or not (roi_x1 <= face_center_x <= roi_x2 and roi_y1 <= face_center_y <= roi_y2):
                cv2.putText(frame, "Please move closer and align face", (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue

            # --- ตรวจจับการยิ้ม ---
            try:
                # ใช้ landmark 61 และ 291 สำหรับปาก
                left_mouth = face_landmarks.landmark[61]
                right_mouth = face_landmarks.landmark[291]
                pt_left_mouth = np.array([left_mouth.x * w, left_mouth.y * h])
                pt_right_mouth = np.array([right_mouth.x * w, right_mouth.y * h])
                mouth_width = np.linalg.norm(pt_left_mouth - pt_right_mouth)
                # ประมาณความกว้างใบหน้าจากแก้มซ้ายและขวา (landmark 234 และ 454)
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
                        cv2.putText(frame, triggered_behavior, (0, 400), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 0), 3)
            except Exception as e:
                print("Smile detection error:", e)

            # --- ตรวจจับการกระพริบตา (Blink) สำหรับยืนยันตัวตน ---
            if not verification_success:
                try:
                    # ใช้ landmark ของดวงตาซ้าย (top: 159, bottom: 145)
                    left_eye_top = face_landmarks.landmark[159]
                    left_eye_bottom = face_landmarks.landmark[145]
                    pt_top = np.array([left_eye_top.x * w, left_eye_top.y * h])
                    pt_bottom = np.array([left_eye_bottom.x * w, left_eye_bottom.y * h])
                    eye_distance = np.linalg.norm(pt_top - pt_bottom)
                    
                    if eye_distance < BLINK_THRESHOLD:
                        if not blink_in_progress:
                            blink_in_progress = True
                            cv2.putText(frame, "BlinkO_O", (0, 450),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            # blink_count += 1
                            # triggered_behavior = f"Blink {blink_count}! (>_<)"
                            # behavior_timestamp = current_time
                            # if blink_count >= 3:
                            #     verification_success = True
                            #     triggered_behavior = "Identity verification success (O_o)"
                            #     behavior_timestamp = current_time
                    else:
                        blink_in_progress = False
                except Exception as e:
                    print("Blink detection error:", e)

            # --- ตรวจจับการหันหน้าซ้าย/ขวา (Head Turn) ---
            try:
                nose = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                
                # --- Horizontal detection (ซ้าย/ขวา) ---
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

                if head_turn_behavior is not None and (current_time - head_turn_timestamp) < 1:
                    cv2.putText(frame, head_turn_behavior, (0, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 3)
                
                # --- Vertical detection (ขึ้น/ก้ม) ---
                left_eye_y = left_eye.y * h
                right_eye_y = right_eye.y * h
                eyes_mid_y = (left_eye_y + right_eye_y) / 2
                nose_y = nose.y * h
                threshold_vertical = 30  # ปรับค่าตามความเหมาะสม

                if (nose_y - eyes_mid_y) > threshold_vertical:
                    vertical = "Turn Down"
                # elif (eyes_mid_y - nose_y) > threshold_vertical:
                #     vertical = "Turn Up"
                else:
                    vertical = "Turn Up"  # หรือสามารถไม่แสดงอะไรก็ได้

                # ตัวอย่างการแสดงผล (ถ้าต้องการแสดงผลในเฟรม)
                cv2.putText(frame, vertical, (0, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 3)

            except Exception as e:
                print("Head turn detection error:", e)


            # --- Head Pose Estimation: คำนวณและวาดแกน X, Y, Z พร้อมแสดง Euler angles ---
            try:
                head_axis, (pitch, yaw, roll), rvec, tvec = estimate_head_pose(face_landmarks.landmark, w, h)
                if head_axis is not None:
                    # ใช้ landmark 1 (จมูก) เป็น origin ของแกน
                    nose_tip = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))
                    pt_x = (int(head_axis[0][0][0]), int(head_axis[0][0][1]))
                    pt_y = (int(head_axis[1][0][0]), int(head_axis[1][0][1]))
                    pt_z = (int(head_axis[2][0][0]), int(head_axis[2][0][1]))
                    cv2.line(frame, nose_tip, pt_x, (0, 0, 255), 3)   # X-axis (แดง)
                    cv2.line(frame, nose_tip, pt_y, (0, 255, 0), 3)   # Y-axis (เขียว)
                    cv2.line(frame, nose_tip, pt_z, (255, 0, 0), 3)   # Z-axis (น้ำเงิน)
                    
                    angle_text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
                    cv2.putText(frame, angle_text, (0, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "Head Pose Error", (0, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
            except Exception as e:
                print("Head pose estimation error:", e)

            # (Optionally) วาด landmark ทั้งหมดบนใบหน้า
            # mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            #                           landmark_drawing_spec=None,
            #                           connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
    else:
        # เมื่อไม่พบใบหน้าในเฟรม ให้รีเซ็ตสถานะพฤติกรรม
        blink_count = 0
        blink_in_progress = False
        # verification_success = False
        triggered_behavior = None
        head_turn_behavior = None
        behavior_timestamp = 0
        head_turn_timestamp = 0

    # --- ตรวจจับมือ (Hand Detection) ---
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Hand Detected", (0, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 128, 255), 2)

    # --- แสดงข้อความเพิ่มเติมบนเฟรม ---
    # if not verification_success:
    #     cv2.putText(frame, "Please blink 3 times slowly", (0, 450),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # if triggered_behavior is not None and (current_time - behavior_timestamp) < 1:
    #     cv2.putText(frame, triggered_behavior, (0, 400), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (0, 0, 0), 3)

    cv2.imshow("Full Face Recognition & Behavior Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
