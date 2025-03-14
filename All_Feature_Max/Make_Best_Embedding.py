import os
import cv2
import numpy as np
import pickle
import insightface
import torch
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# โหลดโมเดล FaceAnalysis และกำหนดค่าของคอมพิวเตอร์
ctx_id = 0 if torch.cuda.is_available() else -1  # ใช้ GPU ถ้ามี
app = FaceAnalysis()
"""
ทำไมต้องกำหนดขนาด 640x640?
-ความสม่ำเสมอของ Input: โมเดล deep learning สำหรับการตรวจจับใบหน้ามักจะถูกฝึกด้วยภาพที่มีขนาดคงที่ การใช้ขนาดเดียวกันสำหรับทุกภาพจะช่วยให้การประมวลผลมีความแม่นยำและมีประสิทธิภาพ
-ประสิทธิภาพและความเร็ว: ขนาด 640x640 ถือเป็นการตั้งค่าที่ให้ความสมดุลระหว่างความเร็วในการประมวลผลและความแม่นยำในการตรวจจับใบหน้า
ถ้าใช้ขนาดอื่นจะมีผลอย่างไร?
-ขนาดเล็กลง (เช่น 320x320)
 ประมวลผลเร็วขึ้น (ลดการใช้ GPU/CPU)อาจทำให้ใบหน้าขนาดเล็กในภาพตรวจจับได้ยากขึ้น
-ขนาดใหญ่ขึ้น (เช่น 800x800 หรือ 1024x1024)
 ตรวจจับใบหน้าได้ละเอียดขึ้น โดยเฉพาะใบหน้าที่เล็กหรืออยู่ไกล
ใช้พลังประมวลผลมากขึ้นและทำงานช้าลง
"""
app.prepare(ctx_id=ctx_id, det_size=(640, 640))

# ฟังก์ชันดึง embeddings จากภาพ
def extract_embeddings(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error loading image {image_path}")
        return []
    
    faces = app.get(img)
    embeddings = [face.normed_embedding for face in faces]
    
    return embeddings if embeddings else []

# เตรียม dataset และดึง embeddings
def prepare_dataset(dataset_path):
    data = {}
    
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            embeddings_list = []
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                embeds = extract_embeddings(image_path)
                if embeds:
                    embeddings_list.extend(embeds)
            if embeddings_list:
                data[folder_name] = np.array(embeddings_list)
    
    return data

# ฟังก์ชันบันทึก embeddings
def save_embeddings(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# ฟังก์ชันโหลด embeddings
def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ฟังก์ชันจับคู่ใบหน้ากับฐานข้อมูล
"""
-new_face_embedding	(Input) ข้อมูล embedding ของใบหน้าที่ต้องการตรวจสอบ (อาร์เรย์ 1 มิติ)
-data	(Input) Dictionary ที่เก็บ embeddings ของบุคคลในฐานข้อมูล {ชื่อคน: embeddings}
-threshold	(Input, Default = 0.4) ค่าความคล้ายคลึงต่ำสุดที่ยอมรับได้ (Cosine Similarity)
-best_match	เก็บชื่อของบุคคลที่มีค่า similarity สูงสุดที่ผ่าน threshold
-best_score	ค่าความคล้ายคลึง (Cosine Similarity) สูงสุดที่พบ
-embeddings	อาร์เรย์ของ embeddings ของบุคคลแต่ละคนในฐานข้อมูล
"""
def match_face(new_face_embedding, data, threshold=0.4):
    if not isinstance(new_face_embedding, np.ndarray):
        new_face_embedding = np.array(new_face_embedding)
    
    if new_face_embedding.size == 0:
        return None, 0  # ไม่มีใบหน้า
    
    best_match = None
    best_score = -1
    new_face_embedding = new_face_embedding.reshape(1, -1)#เเปลง 1D  Ex. [0.1,0.2,0.3,0.4] --> 2D Ex. [[0.1,0.2,0.3,0.4]] ก่อนจะใช้ cosine_similarity(ต้องการ2D )
    for person, embeddings in data.items():
        embeddings = embeddings.reshape(embeddings.shape[0], -1) # ทำให่เเน่ใจว่ายังเป็น 2D อยู่
        scores = cosine_similarity(new_face_embedding, embeddings)
        max_score = max(scores[0])
        if max_score > best_score and max_score > threshold:
            best_match = person
            best_score = max_score
    
    return best_match, best_score


# กำหนด path ของ dataset
dataset_path = 'T:/ML_Learning/ailia-models/face_detection/retinaface/project/knn_examples/Face_Recog/knowns_faces/'

# เตรียมฐานข้อมูล
data = prepare_dataset(dataset_path)
save_embeddings(data, 'embeddings_Best.pkl')

# โหลด embeddings
embeddings = load_embeddings('embeddings_Best.pkl')

# ทดลองทำนายใบหน้าใหม่
new_image_path = 'T:/ML_Learning/ailia-models/face_detection/retinaface/project/knn_examples/Face_Recog/Max.jpg'
new_face_embeddings = extract_embeddings(new_image_path)

if new_face_embeddings:
    result, confidence = match_face(new_face_embeddings[0], embeddings)
    if result:
        print(f"Prediction: {result} (Confidence: {confidence:.2f})")
    else:
        print("No match found (Low confidence)")
else:
    print("No face detected or image is invalid.")
