from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
import shutil
import uuid
import os

app = FastAPI()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def preload_model():
    # Preload model to avoid download delay at first request
    DeepFace.build_model("SFace")

@app.post("/compare-faces/")
async def compare_faces(
    img1: UploadFile,
    img2: UploadFile,
    employee_id: str = Form(...)
):
    try:
        # Save uploaded files
        img1_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{img1.filename}")
        img2_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{img2.filename}")

        with open(img1_path, "wb") as f:
            shutil.copyfileobj(img1.file, f)
        with open(img2_path, "wb") as f:
            shutil.copyfileobj(img2.file, f)

        # Perform face comparison
        result = DeepFace.verify(img1_path, img2_path, model_name="SFace", enforce_detection=True)
        distance = result["distance"]
        threshold = result["threshold"]
        verified = result["verified"]

        similarity = max(0, (1 - (distance / threshold)) * 100)
        similarity = min(similarity, 100)

        # Clean up
        os.remove(img1_path)
        os.remove(img2_path)

        return JSONResponse({
            "employeeId": employee_id,
            "Match": verified,
            "Similarity_Score": f"{similarity:.2f}%",
            "Distance": round(distance, 4),
            "Threshold": round(threshold, 4)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
