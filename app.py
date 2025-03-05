import cv2
import numpy as np
import pickle
import time
import base64
from flask import Flask, render_template, request, jsonify
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp

app = Flask(__name__)

# ============================
# เตรียมโมเดลและตัวแปรพื้นฐาน
# ============================

# Initialize InsightFace FaceAnalysis (ArcFace)
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ใช้ GPU: ctx_id=0, CPU: ctx_id=-1

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize Mediapipe Hands (สำหรับตรวจจับมือ)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ============================
# โหลด Embeddings สำหรับ Face Recognition
# ============================
def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

embeddings_data = load_embeddings('embeddings_Best.pkl')

def extract_embeddings(image):
    faces = face_app.get(image)
    embeddings = []
    bboxes = []
    for face in faces:
        embeddings.append(face.normed_embedding)
        bboxes.append(face.bbox)
    return embeddings, bboxes

def match_face(new_face_embedding, data):
    if new_face_embedding.size == 0:
        return None, 0
    best_match = None
    best_score = -1
    new_face_embedding = np.array(new_face_embedding).reshape(1, -1)
    for person, emb in data.items():
        emb = emb.reshape(emb.shape[0], -1)
        scores = cosine_similarity(new_face_embedding, emb)
        max_score = max(scores[0])
        if max_score > best_score:
            best_match = person
            best_score = max_score
    return best_match, best_score

# ============================
# กำหนดฟังก์ชันสำหรับแต่ละ Feature
# ============================

def process_face_recognition(frame):
    embeddings, _ = extract_embeddings(frame)  # ลบ bboxes ทิ้ง
    results = []
    if embeddings:
        for embedding in embeddings:
            result, score = match_face(embedding, embeddings_data)
            results.append({
                "result": result if result else "No match found",
                "score": float(score)  # เพิ่มคะแนนความเหมือน
            })
    return results



def process_smile_detection(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    detected = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            try:
                left_mouth = face_landmarks.landmark[61]
                right_mouth = face_landmarks.landmark[291]
                pt_left = np.array([left_mouth.x * w, left_mouth.y * h])
                pt_right = np.array([right_mouth.x * w, right_mouth.y * h])
                mouth_width = np.linalg.norm(pt_left - pt_right)
                # ประมาณความกว้างใบหน้าจากแก้ม (landmark 234 และ 454)
                left_face = face_landmarks.landmark[234]
                right_face = face_landmarks.landmark[454]
                pt_left_face = np.array([left_face.x * w, left_face.y * h])
                pt_right_face = np.array([right_face.x * w, right_face.y * h])
                face_width = np.linalg.norm(pt_left_face - pt_right_face)
                if face_width > 0:
                    smile_ratio = mouth_width / face_width
                    if smile_ratio > 0.5:
                        detected = True
                        break
            except Exception as e:
                print("Smile detection error:", e)
    return detected

# กำหนดค่าตัวแปรสำหรับ Blink Detection
BLINK_THRESHOLD = 5

def process_blink_detection(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    blink_detected = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]
                h, w, _ = frame.shape
                pt_top = np.array([left_eye_top.x * w, left_eye_top.y * h])
                pt_bottom = np.array([left_eye_bottom.x * w, left_eye_bottom.y * h])
                eye_distance = np.linalg.norm(pt_top - pt_bottom)
                if eye_distance < BLINK_THRESHOLD:
                    blink_detected = True
                    break
            except Exception as e:
                print("Blink detection error:", e)
    return blink_detected

def process_head_turn_detection(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                h, w, _ = frame.shape
                # ใช้ landmark สำหรับจมูกและตา
                nose = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                
                # --- Horizontal detection (ซ้าย/ขวา) --- 
                nose_x = nose.x * w
                left_eye_x = left_eye.x * w
                right_eye_x = right_eye.x * w
                eyes_mid_x = (left_eye_x + right_eye_x) / 2
                horizontal_diff = eyes_mid_x - nose_x
                threshold_horizontal = 15  # ปรับค่าตามความเหมาะสม
                if horizontal_diff > threshold_horizontal:
                    horizontal = "Turn Right-->"
                elif horizontal_diff < -threshold_horizontal:
                    horizontal = "Turn Left<--"
                else:
                    horizontal = "No horizontal turn"
                
                # --- Vertical detection (ขึ้น/ก้ม) --- 
                left_eye_y = left_eye.y * h
                right_eye_y = right_eye.y * h
                eyes_mid_y = (left_eye_y + right_eye_y) / 2
                nose_y = nose.y * h
                vertical_diff = nose_y - eyes_mid_y
                threshold_vertical = 25  # ปรับค่าตามความเหมาะสม
                if vertical_diff > threshold_vertical:
                    vertical = "Turn Down"
                elif vertical_diff < threshold_vertical:
                    vertical = "Turn Up"
                else:
                    vertical = "No vertical turn"
                
                # รวมผลลัพธ์ทั้งสองเข้าด้วยกัน
                return f"{horizontal} | {vertical}"
            except Exception as e:
                print("Head turn detection error:", e)
    return "No head turn"


def estimate_head_pose(landmarks, w, h):
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w, landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h)
    ], dtype="double")
    
    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1]
    ])
    
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None, None
    axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
    imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = euler_angles.flatten()
    return imgpts, (pitch, yaw, roll), rotation_vector, translation_vector

def process_head_pose_estimation(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                h, w, _ = frame.shape
                head_axis, angles, rvec, tvec = estimate_head_pose(face_landmarks.landmark, w, h)
                if head_axis is not None:
                    pitch, yaw, roll = angles
                    return {"pitch": pitch, "yaw": yaw, "roll": roll}
            except Exception as e:
                print("Head pose estimation error:", e)
    return {}



# ============================
# API Endpoint สำหรับประมวลผล Frame
# ============================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    img_data = data.get('image')
    feature = data.get('feature')
    
    # แปลง Base64 image เป็น numpy array
    try:
        img_str = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": "Error processing image", "details": str(e)})
    
    # เลือกประมวลผลตาม Feature ที่ร้องขอ
    result = None
    if feature == "Face Recognition":
        result = process_face_recognition(frame)
    elif feature == "Smile Detection":
        result = process_smile_detection(frame)
    elif feature == "Blink Detection":
        result = process_blink_detection(frame)
    elif feature == "Head Turn Detection":
        result = process_head_turn_detection(frame)
    elif feature == "Head Pose Estimation":
        result = process_head_pose_estimation(frame)
    else:
        result = "Unknown feature"
    
    # จัดรูปแบบผลลัพธ์ให้อยู่ในรูปแบบ JSON ที่เรียบง่าย
    response = {}
    if feature == "Face Recognition":
        response["face_recognition"] = result
    elif feature == "Smile Detection":
        response["smile_detected"] = result
    elif feature == "Blink Detection":
        response["blink_detected"] = result
    elif feature == "Head Turn Detection":
        response["head_turn"] = result
    elif feature == "Head Pose Estimation":
        response["head_pose"] = result
    else:
        response["error"] = result
    
    return jsonify(response)

# ฟังก์ชันสำหรับดึง embedding จาก frame (สำหรับ compare face)
def get_face_embedding_from_frame(frame):
    faces = face_app.get(frame)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    # หากมี keypoints สามารถจัดแนวใบหน้าได้ (เพิ่มเติมได้ตามต้องการ)
    # เราจะใช้ embedding ที่ได้จาก face.normed_embedding
    return face.normed_embedding, face

# ตั้งค่าค่าคล้ายกัน (threshold) สำหรับการเปรียบเทียบใบหน้า
FACE_COMPARISON_THRESHOLD = 0.6

@app.route('/compareface', methods=["GET", "POST"])
def compare_face():
    if request.method == "POST":
        # รับไฟล์จาก form
        file1 = request.files.get("image1")
        file2 = request.files.get("image2")
        if not file1 or not file2:
            return jsonify({"error": "กรุณาอัปโหลดไฟล์ทั้งสองภาพ"}), 400

        # อ่านไฟล์เป็น numpy array โดยไม่ต้องบันทึกลงดิสก์
        npimg1 = np.frombuffer(file1.read(), np.uint8)
        npimg2 = np.frombuffer(file2.read(), np.uint8)
        img1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)

        # ตรวจสอบว่ารูปโหลดได้หรือไม่
        if img1 is None or img2 is None:
            return jsonify({"error": "ไม่สามารถประมวลผลภาพที่อัปโหลดได้"}), 400

        # ดึง embedding ของแต่ละภาพ
        emb1, face1 = get_face_embedding_from_frame(img1)
        emb2, face2 = get_face_embedding_from_frame(img2)
        if emb1 is None or emb2 is None:
            return jsonify({"error": "ไม่พบใบหน้าในหนึ่งในภาพ"}), 400

        # คำนวณ cosine similarity ระหว่าง embedding ทั้งสอง
        similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        result_text = "✅ Same person" if similarity > FACE_COMPARISON_THRESHOLD else "❌ Different person"

        # ส่งผลลัพธ์ไปยัง template สำหรับแสดงผล
        return render_template('compare_face_result.html', similarity=similarity, result_text=result_text)

    # เมื่อเป็น GET ให้แสดงฟอร์มอัปโหลด
    return render_template('compare_face.html')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare_face_page')
def compare_face_page():
    return render_template('compare_face.html')

if __name__ == '__main__':
    app.run(debug=True)
