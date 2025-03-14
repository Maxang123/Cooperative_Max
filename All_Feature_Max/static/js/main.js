document.addEventListener("DOMContentLoaded", function () {
  const video = document.getElementById("video");
  const openCamBtn = document.getElementById("OpenCamera");
  const closeCamBtn = document.getElementById("CloseCamera");
  const resultsContainer = document.getElementById("results");

  let stream = null;
  let intervals = {};

  // ฟังก์ชันเปิดกล้อง
  openCamBtn.addEventListener("click", async function () {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      openCamBtn.style.display = "none";
      closeCamBtn.style.display = "inline-block";
      document.getElementById("controls").style.display = "block";
    } catch (error) {
      console.error("เปิดกล้องไม่สำเร็จ:", error);
    }
  });

  // ฟังก์ชันปิดกล้อง
  closeCamBtn.addEventListener("click", function () {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      video.srcObject = null;
      stream = null;
    }
    openCamBtn.style.display = "inline-block";
    closeCamBtn.style.display = "none";
  });

  // ฟังก์ชันสำหรับ capture frame และส่งไปประมวลผล
  function processFrame(feature) {
    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    let imageData = canvas.toDataURL("image/jpeg");

    fetch("/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData, feature: feature })
    })
      .then(response => response.json())
      .then(data => {
        let resultId = "result_" + feature.replace(/\s+/g, "");
        let resultElement = document.getElementById(resultId);
        if (!resultElement) {
          resultElement = document.createElement("div");
          resultElement.id = resultId;
          resultElement.style.marginTop = "10px";
          resultElement.style.fontSize = "16px";
          resultsContainer.appendChild(resultElement);
        }
        let displayText = "";
        if (feature === "Face Recognition") {
          if (data.face_recognition && Array.isArray(data.face_recognition) && data.face_recognition.length > 0) {
            let face = data.face_recognition[0];
            displayText = "Face Recognition: " + face.result + " (Score: " + face.score.toFixed(2) + ")";
          } else {
            displayText = "Face Recognition: No face detected";
          }
        }
        
         else if (feature === "Smile Detection") {
          displayText = "Smile Detection: " + (data.smile_detected ? "Smile" : "No Smile");
        } else if (feature === "Blink Detection") {
          displayText = "Blink Detection: " + (data.blink_detected ? "Blink" : "No Blink");
        } else if (feature === "Head Turn Detection") {
          displayText = "Head Turn: " + data.head_turn;
        } else if (feature === "Head Pose Estimation") {
          if (data.head_pose && Object.keys(data.head_pose).length > 0) {
            displayText = `Head Pose: Pitch: ${data.head_pose.pitch.toFixed(1)}, Yaw: ${data.head_pose.yaw.toFixed(1)}, Roll: ${data.head_pose.roll.toFixed(1)}`;
          } else {
            displayText = "Head Pose: Not Detected";
          }
        }
        else if (feature === "Glasses Detection") {
          displayText = "Glasses Detection: " + (data.glasses_detected ? "Glasses" : "No Glasses");
        }
        else if (feature === "Mask Detection") {
          displayText = "Mask Detection: " + (data.mask_detected ? "Mask Detected" : "No Mask Detected");
        }
        else if (feature === "Anti Spoofing Detection") {
          displayText = "Anti Spoofing: " + data.anti_spoofing;
        }
        
        
        resultElement.innerHTML = displayText;
      })
      .catch(error => console.error("Error:", error));
  }

  // ฟังก์ชันสำหรับ toggle Feature เมื่อ checkbox เปลี่ยนสถานะ
  function toggleFeature(checkboxId, featureName) {
    const checkbox = document.getElementById(checkboxId);
    checkbox.addEventListener("change", function () {
      if (this.checked) {
        intervals[featureName] = setInterval(() => {
          processFrame(featureName);
        }, 1000);
      } else {
        clearInterval(intervals[featureName]);
        let resultElement = document.getElementById("result_" + featureName.replace(/\s+/g, ""));
        if (resultElement) resultElement.innerHTML = "";
      }
    });
  }

  toggleFeature("faceRecognition", "Face Recognition");
  toggleFeature("smileDetection", "Smile Detection");
  toggleFeature("blinkDetection", "Blink Detection");
  toggleFeature("headTurnDetection", "Head Turn Detection");
  toggleFeature("headPoseEstimation", "Head Pose Estimation");
  toggleFeature("glassesDetection", "Glasses Detection");
  toggleFeature("maskDetection", "Mask Detection");
  toggleFeature("antiSpoofingDetection", "Anti Spoofing Detection");



});
