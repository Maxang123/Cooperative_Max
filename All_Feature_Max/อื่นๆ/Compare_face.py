import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# โหลดโมเดล ArcFace
app = FaceAnalysis(name="buffalo_l")  # โมเดลที่มีความแม่นยำสูง
app.prepare(ctx_id=-1, det_size=(640, 640))  # ใช้ GPU (ถ้ามี) -1 ใช้ CPU

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: ไม่สามารถโหลดภาพได้จาก {image_path}")
        return None, None

    faces = app.get(img)
    
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None, None
    
    face = faces[0]
    
    # ตรวจสอบว่ามี keypoints สำหรับจัดแนวใบหน้าหรือไม่
    if hasattr(face, 'kps'):
        try:
            # ใช้ keypoints ในการจัดแนวใบหน้า โดยใช้ norm_crop จาก insightface
            from insightface.utils import face_align
            aligned_face = face_align.norm_crop(img, face.kps)
            # คำนวณ embedding จากใบหน้าที่จัดแนวแล้ว
            embedding = app.model.get_embedding(aligned_face)
            # อัปเดต embedding ใน object face (ถ้าต้องการใช้งานต่อ)
            face.embedding = embedding
        except Exception as e:
            print(f"Face alignment failed: {e}")
            # หากจัดแนวไม่สำเร็จ จะใช้ embedding เดิมที่ได้จากการตรวจจับ
    return face.embedding, faces

def display_faces(image_path, faces, window_name, extra_text=""):
    """
    ฟังก์ชันสำหรับวาดกรอบใบหน้าบนภาพและแสดงผล
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: ไม่สามารถโหลดภาพสำหรับแสดงผลจาก {image_path}")
        return

    # วาดกรอบใบหน้า
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # แสดงคะแนนการตรวจจับ (ถ้ามี)
        if hasattr(face, 'det_score'):
            cv2.putText(img, f"{face.det_score:.2f}", (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # ถ้ามีข้อความเพิ่มเติมให้แสดงที่มุมบนซ้าย
    if extra_text:
        cv2.putText(img, extra_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow(window_name, img)

def compare_faces(img1_path, img2_path, threshold=0.65):
    emb1, faces1 = get_face_embedding(img1_path)
    emb2, faces2 = get_face_embedding(img2_path)
    
    if emb1 is None or emb2 is None:
        print("Face not found in one of the images.")
        return None

    # คำนวณ cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"Similarity Score: {similarity:.4f}")
    
    result_text = "✅ Same person" if similarity > threshold else "❌ Different person"
    print(result_text)
    
    # แสดงภาพพร้อมกรอบใบหน้าพร้อมข้อความผลการตรวจสอบ
    display_faces(img1_path, faces1, "Image 1", extra_text=result_text)
    display_faces(img2_path, faces2, "Image 2", extra_text=result_text)
    
    # รอให้ผู้ใช้กดปุ่มเพื่อปิดหน้าต่างภาพ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return similarity > threshold

# ใส่ path ของภาพที่ต้องการเปรียบเทียบ
img1 = "T:/ML_Learning/ailia-models/face_detection/retinaface/project/knn_examples/Face_Recog/beam.jpeg"
img2 = "T:/ML_Learning/ailia-models/face_detection/retinaface/project/knn_examples/Face_Recog/kawee.jpeg"
compare_faces(img1, img2)
